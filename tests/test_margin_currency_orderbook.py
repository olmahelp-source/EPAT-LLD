import math
import unittest
from datetime import datetime

import pandas as pd

import helpers  # noqa: F401
from algoplatform.trading.brokers import MarketSimulator, OrderBook, ticks_to_bars
from algoplatform.foundation.core.enums import OrderType, Side, TimeInForce
from algoplatform.foundation.core.exceptions import MissingDataError, OrderError, OrderNotFoundError, ValidationError
from algoplatform.foundation.core.models import Fill, Instrument, Order
from algoplatform.market_data.data import MarketData
from algoplatform.market_data.instruments import future_spec, option_spec
from algoplatform.capital.portfolio import (CashLedger, FxRates, Portfolio, exposure_by_currency, fx_pnl_attribution, hedge_notionals,
                                    portfolio_currency_exposure)
from algoplatform.quantitative.quant import bs_price
from algoplatform.capital.risk import (MarginCheck, MarginPosition, MarginRules, RiskContext, ScanSettings, linear_requirement,
                               liquidation_price, margin_report, position_margin, rule_based_margin, scenario_margin)


def option(qty, strike, right, price=None, spot=100.0, vol=0.25, t=0.25, symbol=None):
    px = bs_price(spot, strike, t, 0.0, vol, right) if price is None else price
    return MarginPosition(symbol or f"{right[0].upper()}{strike:g}", qty, px, "option", 100, underlying="X", underlying_price=spot,
                          strike=strike, right=right, years_to_expiry=t, volatility=vol)


STOCK = MarginPosition("X", 100, 100.0)


class RuleBasedMarginTests(unittest.TestCase):
    def test_equities_long_and_short(self):
        self.assertEqual(position_margin(MarginPosition("AAA", 100, 50.0)), (2500.0, 1250.0))        # 50% to open, 25% to maintain
        self.assertEqual(position_margin(MarginPosition("AAA", -100, 50.0)), (2500.0, 1500.0))       # shorts: 30% maintenance
        self.assertEqual(position_margin(MarginPosition("AAA", 0, 50.0)), (0.0, 0.0))
        custom = MarginRules(equity_initial_long=1.0, equity_maintenance_long=1.0)                  # a cash account: no borrowing
        self.assertEqual(position_margin(MarginPosition("AAA", 100, 50.0), custom), (5000.0, 5000.0))

    def test_futures_use_the_contract_margin_fraction(self):
        fut = MarginPosition("ESH5", -2, 5000.0, "future", multiplier=50, margin_pct=0.06)
        initial, maintenance = position_margin(fut)
        self.assertAlmostEqual(initial, 2 * 5000 * 50 * 0.06)
        self.assertAlmostEqual(maintenance, initial * 0.8)

    def test_options(self):
        long_call = option(2, 105, "call", price=3.0)
        self.assertEqual(position_margin(long_call), (600.0, 600.0))                  # paid in full, nothing more
        short_otm_call = option(-1, 105, "call", price=3.0)
        self.assertAlmostEqual(position_margin(short_otm_call)[0], (3.0 + 0.20 * 100 - 5.0) * 100)
        deep_otm = option(-1, 140, "call", price=0.1)
        self.assertAlmostEqual(position_margin(deep_otm)[0], (0.1 + 0.10 * 100) * 100)               # the 10% floor applies
        short_put = option(-1, 90, "put", price=1.0)
        self.assertAlmostEqual(position_margin(short_put)[0], (1.0 + 0.20 * 100 - 10.0) * 100)
        self.assertAlmostEqual(position_margin(option(-1, 60, "put", price=0.05))[0], (0.05 + 0.10 * 60) * 100)   # put floor is on the strike

    def test_portfolio_sum_and_no_relief_for_hedges(self):
        pos = [MarginPosition("AAA", 100, 50.0), MarginPosition("BBB", -100, 50.0)]
        total = rule_based_margin(pos)
        self.assertEqual((total["initial"], total["maintenance"]), (5000.0, 2750.0))
        self.assertEqual(total["by_symbol"]["BBB"], {"initial": 2500.0, "maintenance": 1500.0})

    def test_validation(self):
        for kwargs in ({"symbol": "A", "quantity": 1, "price": -1.0}, {"symbol": "A", "quantity": float("nan"), "price": 1.0},
                       {"symbol": "A", "quantity": 1, "price": 1.0, "asset_class": "nft"}, {"symbol": "A", "quantity": 1, "price": 1.0, "multiplier": 0},
                       {"symbol": "O", "quantity": 1, "price": 1.0, "asset_class": "option", "strike": 100, "right": "call"},
                       {"symbol": "O", "quantity": 1, "price": 1.0, "asset_class": "option", "underlying_price": 100.0, "right": "both", "strike": 1}):
            with self.assertRaises(ValidationError, msg=str(kwargs)):
                MarginPosition(**kwargs)
        with self.assertRaises(ValidationError):
            MarginRules(equity_initial_long=0)

    def test_built_from_a_contract_record(self):
        fut = MarginPosition.from_spec(future_spec("ES", "2025-03-21", 50, 0.25, margin_pct=0.06).to_dict(), 2, 5000.0)
        self.assertEqual((fut.asset_class, fut.multiplier, fut.margin_pct, fut.group), ("future", 50.0, 0.06, "ES"))
        opt = MarginPosition.from_spec(option_spec("AAPL", "2025-01-17", "put", 150).to_dict(), -1, 4.0, days_to_expiry=73,
                                       underlying_price=160.0, volatility=0.3)
        self.assertEqual((opt.right, opt.strike, opt.group), ("put", 150.0, "AAPL"))
        self.assertAlmostEqual(opt.years_to_expiry, 0.2)


class ScenarioMarginTests(unittest.TestCase):
    def test_linear_position_is_charged_the_scan_range(self):
        out = scenario_margin([STOCK], ScanSettings(price_scan=0.15, extreme_multiple=1.0))
        self.assertAlmostEqual(out["maintenance"], 100 * 100 * 0.15)
        self.assertAlmostEqual(out["initial"], out["maintenance"] * 1.10)
        self.assertEqual(out["by_underlying"]["X"]["worst_price_move"], -0.15)
        short = scenario_margin([MarginPosition("X", -100, 100.0)], ScanSettings(price_scan=0.15, extreme_multiple=1.0))
        self.assertEqual(short["by_underlying"]["X"]["worst_price_move"], 0.15)       # a short loses when the price rises

    def test_hedges_reduce_the_requirement(self):
        naked = scenario_margin([option(-1, 100, "call")])["maintenance"]
        spread = scenario_margin([option(-1, 100, "call"), option(1, 105, "call")])["maintenance"]
        self.assertLess(spread, 0.5 * naked)                                          # the long call caps the loss
        self.assertLessEqual(spread, 500.0 + 1e-6)                                    # never more than the width of the spread
        covered = scenario_margin([option(-1, 105, "call"), STOCK])["maintenance"]
        self.assertLess(covered, scenario_margin([STOCK])["maintenance"])             # the premium cushions the stock's downside
        protected = scenario_margin([STOCK, option(1, 95, "put")])["maintenance"]
        self.assertLess(protected, scenario_margin([STOCK])["maintenance"])           # a protective put floors it
        rules_spread = rule_based_margin([option(-1, 100, "call"), option(1, 105, "call")])["initial"]
        self.assertGreater(rules_spread, spread)                                      # the rule-based method gives the hedge no credit

    def test_fully_hedged_book_needs_almost_nothing_but_shorts_keep_a_floor(self):
        flat = scenario_margin([STOCK, MarginPosition("X_FUT", -100, 100.0, underlying="X")])
        self.assertAlmostEqual(flat["maintenance"], 0.0)
        far = scenario_margin([option(-1, 200, "call", t=0.02)])                      # practically worthless short call
        info = far["by_underlying"]["X"]
        self.assertAlmostEqual(info["short_option_minimum"], 100 * 100 * 0.005)
        self.assertGreaterEqual(far["maintenance"], info["short_option_minimum"])     # never zero for a short option

    def test_volatility_and_extreme_scenarios_matter_for_short_options(self):
        short_strangle = [option(-1, 85, "put"), option(-1, 115, "call")]
        calm = scenario_margin(short_strangle, ScanSettings(vol_scan=0.0, extreme_multiple=1.0))["maintenance"]
        with_vol = scenario_margin(short_strangle, ScanSettings(vol_scan=0.5, extreme_multiple=1.0))["maintenance"]
        with_extreme = scenario_margin(short_strangle, ScanSettings(vol_scan=0.0, extreme_multiple=3.0))["maintenance"]
        self.assertGreater(with_vol, calm)
        self.assertGreater(with_extreme, calm)

    def test_no_offset_across_underlyings_and_validation(self):
        a, b = MarginPosition("A", 100, 100.0), MarginPosition("B", -100, 100.0)
        both = scenario_margin([a, b])
        self.assertAlmostEqual(both["maintenance"], scenario_margin([a])["maintenance"] + scenario_margin([b])["maintenance"])
        self.assertEqual(set(both["by_underlying"]), {"A", "B"})
        self.assertEqual(scenario_margin([])["maintenance"], 0.0)
        bad = MarginPosition("O", -1, 1.0, "option", 100, underlying="X", underlying_price=100.0, strike=100.0, right="call")
        with self.assertRaises(ValidationError):
            scenario_margin([bad])                                                    # no volatility / expiry to reprice with
        for kwargs in ({"price_scan": 0}, {"steps": 0}, {"extreme_cover": 0}, {"initial_ratio": 0.5}):
            with self.assertRaises(ValidationError):
                ScanSettings(**kwargs)
        self.assertEqual(len(ScanSettings(steps=3, vol_scan=0.25).scenarios()), 7 * 2 + 2)


class MarginAccountTests(unittest.TestCase):
    def test_report_and_margin_call(self):
        pos = [MarginPosition("AAA", 1000, 50.0)]
        ok = margin_report(pos, equity=30_000)
        self.assertEqual((ok["initial"], ok["maintenance"], ok["available_funds"], ok["excess_liquidity"]), (25_000, 12_500, 5_000, 17_500))
        self.assertFalse(ok["margin_call"])
        self.assertAlmostEqual(ok["utilisation"], 25 / 30)
        self.assertAlmostEqual(ok["leverage"], 50 / 30)
        call = margin_report(pos, equity=10_000)
        self.assertTrue(call["margin_call"])
        self.assertEqual(call["shortfall"], 2_500)
        scen = margin_report([option(-1, 100, "call"), option(1, 105, "call")], 2_000, method="scenario")
        self.assertEqual(scen["method"], "scenario")
        self.assertFalse(scen["margin_call"])
        with self.assertRaises(ValidationError):
            margin_report(pos, 1.0, method="guess")
        self.assertEqual(margin_report(pos, 0.0)["utilisation"], math.inf)

    def test_liquidation_price_is_where_equity_meets_maintenance(self):
        qty, entry, cash_equity, mm = 1000, 50.0, 25_000.0, 0.25
        p = liquidation_price(qty, entry, cash_equity, mm)
        self.assertAlmostEqual(cash_equity + qty * (p - entry), mm * qty * p)         # the defining equation holds
        self.assertLess(p, entry)
        ps = liquidation_price(-qty, entry, cash_equity, 0.30)
        self.assertAlmostEqual(cash_equity - qty * (ps - entry), 0.30 * qty * ps)
        self.assertGreater(ps, entry)
        self.assertEqual(liquidation_price(100, 50.0, 1e9, 0.25), 0.0)                # so much equity it can never be called
        with self.assertRaises(ValidationError):
            liquidation_price(0, 50.0, 1.0, 0.25)

    def test_pre_trade_check(self):
        check = MarginCheck(linear_requirement(futures={"FUT": 0.10}, multipliers={"FUT": 50}))
        ctx = RiskContext(equity=10_000, cash=10_000, positions={}, prices={"AAA": 100.0, "FUT": 1000.0}, now=datetime(2024, 1, 2))
        buy = lambda sym, qty, side=Side.BUY: Order("o", sym, side, qty)              # noqa: E731
        self.assertIsNone(check.check(buy("AAA", 150), ctx))                          # needs 7,500 of 10,000
        self.assertIn("margin requirement", check.check(buy("AAA", 250), ctx))        # needs 12,500
        self.assertIsNone(check.check(buy("FUT", 2), ctx))                            # 2 x 1000 x 50 x 10% = 10,000: exactly at the limit
        self.assertIsNotNone(check.check(buy("FUT", 3), ctx))
        pending = RiskContext(10_000, 10_000, {}, {"AAA": 100.0}, datetime(2024, 1, 2), pending={"AAA": 150.0})
        self.assertIsNotNone(check.check(buy("AAA", 100), pending))                   # working orders count
        tight = MarginCheck(linear_requirement(), max_utilisation=0.5)
        self.assertIsNotNone(tight.check(buy("AAA", 150), ctx))

    def test_reducing_risk_is_never_blocked_and_unknowns_fail_closed(self):
        check = MarginCheck(linear_requirement())
        over = RiskContext(equity=1_000, cash=0, positions={"AAA": 500.0}, prices={"AAA": 100.0}, now=datetime(2024, 1, 2))   # far over its limit
        self.assertIsNone(check.check(Order("o", "AAA", Side.SELL, 200), over))       # selling down must go through
        self.assertIsNotNone(check.check(Order("o", "AAA", Side.BUY, 1), over))
        no_price = RiskContext(10_000, 10_000, {}, {}, datetime(2024, 1, 2))
        self.assertIn("could not be computed", check.check(Order("o", "ZZZ", Side.BUY, 1), no_price))
        limit = Order("o", "ZZZ", Side.BUY, 10, OrderType.LIMIT, limit_price=50.0)
        self.assertIsNone(check.check(limit, no_price))                               # a limit price is a usable reference
        with self.assertRaises(ValidationError):
            MarginCheck(linear_requirement(), max_utilisation=0)


class FxRateTests(unittest.TestCase):
    def setUp(self):
        self.fx = FxRates().set("EURUSD", 1.10).set("USD/JPY", 150.0).set(("GBP", "USD"), 1.25)

    def test_direct_inverse_and_cross_rates(self):
        self.assertEqual(self.fx.rate("EUR", "USD"), 1.10)
        self.assertAlmostEqual(self.fx.rate("USD", "EUR"), 1 / 1.10)
        self.assertEqual(self.fx.rate("usd", "USD"), 1.0)
        self.assertAlmostEqual(self.fx.rate("EUR", "JPY"), 165.0)                     # through the pivot
        self.assertAlmostEqual(self.fx.rate("JPY", "GBP"), 1 / 150 / 1.25)
        self.assertAlmostEqual(self.fx.convert(1000, "EUR", "GBP"), 1000 * 1.10 / 1.25)
        self.assertAlmostEqual(self.fx.rate("EUR", "JPY") * self.fx.rate("JPY", "EUR"), 1.0)
        self.assertEqual(self.fx.pairs(), ["EURUSD", "GBPUSD", "USDJPY"])

    def test_missing_and_invalid(self):
        with self.assertRaises(MissingDataError) as cm:
            self.fx.rate("EUR", "CHF")
        self.assertIn("EURUSD", cm.exception.details["known_pairs"])
        for bad in ("EURO", "EUR", "EUREUR", "12345X"):
            with self.assertRaises(ValidationError, msg=bad):
                self.fx.set(bad, 1.0)
        for rate in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValidationError):
                self.fx.set("EURUSD", rate)

    def test_point_in_time_lookup(self):
        fx = FxRates()
        fx.add_series("EURUSD", pd.Series([1.05, 1.10, 1.20], index=pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"])))
        self.assertEqual(fx.rate("EUR", "USD", "2024-02-15"), 1.10)                   # the rate known then, not the latest
        self.assertEqual(fx.rate("EUR", "USD", "2024-02-01"), 1.10)
        self.assertEqual(fx.rate("EUR", "USD"), 1.20)
        with self.assertRaises(MissingDataError):
            fx.rate("EUR", "USD", "2023-12-31")                                       # nothing had been published yet
        fx.set("EURUSD", 1.00)                                                        # a standing rate fills the gap before the series
        self.assertEqual(fx.rate("EUR", "USD", "2023-12-31"), 1.00)
        self.assertEqual(fx.rate("EUR", "USD", "2024-01-15"), 1.05)
        fx.set("EURUSD", 1.30, ts="2024-04-01")
        self.assertEqual(fx.rate("EUR", "USD"), 1.30)
        with self.assertRaises(ValidationError):
            fx.add_series("EURUSD", pd.Series([1.0, -1.0], index=pd.to_datetime(["2024-01-01", "2024-01-02"])))


class CashLedgerTests(unittest.TestCase):
    def test_balances_exchange_and_settlement(self):
        cash = CashLedger({"USD": 10_000})
        received = cash.exchange("USD", "EUR", 5_500, rate=1 / 1.10, fee_bps=10)
        self.assertAlmostEqual(received, 5_000 * 0.999)
        self.assertEqual(cash.balance("USD"), 4_500)
        cash.settle("EUR", -3_000, "buy 30 SAP @ 100")
        cash.settle("EUR", -5.0, "commission")
        self.assertAlmostEqual(cash.balance("eur"), 5_000 * 0.999 - 3_005)
        self.assertEqual([e.kind for e in cash.entries], ["deposit", "exchange", "exchange", "settle", "settle"])
        fx = FxRates().set("EURUSD", 1.10)
        self.assertAlmostEqual(cash.total("USD", fx), 4_500 + cash.balance("EUR") * 1.10)
        self.assertEqual(set(cash.balances()), {"EUR", "USD"})

    def test_an_unfunded_purchase_is_an_error_not_a_silent_loan(self):
        cash = CashLedger({"USD": 1_000})
        with self.assertRaises(ValidationError) as cm:
            cash.settle("EUR", -500)
        self.assertEqual(cm.exception.code, "INSUFFICIENT_CASH")
        with self.assertRaises(ValidationError):
            cash.exchange("USD", "EUR", 2_000, 0.9)                                   # more than is held
        self.assertEqual(cash.balances(), {"USD": 1_000.0})                           # the failed exchange changed nothing
        with self.assertRaises(ValidationError):
            cash.withdraw("USD", 1_500)
        margin = CashLedger({"USD": 1_000}, allow_negative=True)
        margin.settle("EUR", -500)
        self.assertEqual(margin.balance("EUR"), -500)
        for call in (lambda: cash.deposit("USD", 0), lambda: cash.withdraw("USD", -1), lambda: cash.exchange("USD", "USD", 1, 1),
                     lambda: cash.exchange("USD", "EUR", 1, 0)):
            with self.assertRaises(ValidationError):
                call()


class ExposureTests(unittest.TestCase):
    def setUp(self):
        self.fx = FxRates().set("EURUSD", 1.10).set("USDJPY", 150.0)

    def test_exposure_by_currency(self):
        out = exposure_by_currency({"USD": 5_000, "EUR": 4_000, "JPY": -300_000}, "USD", self.fx)
        self.assertAlmostEqual(out["total"], 5_000 + 4_400 - 2_000)
        self.assertAlmostEqual(out["gross"], 11_400)
        self.assertAlmostEqual(out["fx_exposure"], 6_400 / 11_400)
        self.assertAlmostEqual(sum(r["share"] for r in out["by_currency"].values()), 1.0)
        self.assertEqual(exposure_by_currency({"USD": 100}, "USD", self.fx)["fx_exposure"], 0.0)
        hedges = hedge_notionals(out, 0.5)
        self.assertEqual(hedges, {"EUR": 2_000.0, "JPY": -150_000.0})                 # sell half the euros, buy back half the yen
        with self.assertRaises(ValidationError):
            hedge_notionals(out, 1.5)
        with self.assertRaises(MissingDataError):
            exposure_by_currency({"CHF": 1}, "USD", self.fx)

    def test_portfolio_ledger_breakdown(self):
        instruments = {"SAP": Instrument("SAP", currency="EUR"), "AAPL": Instrument("AAPL", currency="USD")}
        pf = Portfolio(100_000, instruments)
        ts = datetime(2024, 1, 2)
        pf.apply_fill(Fill("f1", "o1", "SAP", Side.BUY, 100, 100.0, ts))
        pf.apply_fill(Fill("f2", "o2", "AAPL", Side.BUY, 50, 200.0, ts))
        pf.apply_fill(Fill("f3", "o3", "UNLISTED", Side.BUY, 10, 10.0, ts))           # no instrument record: assumed base currency
        pf.mark({"SAP": 110.0, "AAPL": 200.0, "UNLISTED": 10.0}, ts)
        out = portfolio_currency_exposure(pf, self.fx, "USD", cash={"USD": 70_000, "EUR": 5_000})
        self.assertAlmostEqual(out["by_currency"]["EUR"]["local"], 100 * 110 + 5_000)
        self.assertAlmostEqual(out["by_currency"]["USD"]["local"], 50 * 200 + 100 + 70_000)
        self.assertAlmostEqual(out["total"], 16_000 * 1.10 + 80_100)
        single = portfolio_currency_exposure(pf, self.fx, "USD")
        self.assertAlmostEqual(single["by_currency"]["USD"]["local"], 10_100 + pf.cash)

    def test_pnl_attribution_adds_up(self):
        out = fx_pnl_attribution(1_000, 1_100, 1.10, 1.21)                            # asset +10% and currency +10%
        self.assertAlmostEqual(out["asset"], 110.0)
        self.assertAlmostEqual(out["currency"], 110.0)
        self.assertAlmostEqual(out["interaction"], 11.0)
        self.assertAlmostEqual(out["asset"] + out["currency"] + out["interaction"], out["total"])
        flat = fx_pnl_attribution(1_000, 1_000, 1.10, 0.99)                           # the asset did nothing; the currency lost 10%
        self.assertAlmostEqual(flat["total"], -110.0)
        self.assertEqual(flat["asset"], 0.0)
        with self.assertRaises(ValidationError):
            fx_pnl_attribution(1, 1, 0, 1)


def book():
    b = OrderBook(0.01)
    b.add_limit("b1", Side.BUY, 99.99, 100)
    b.add_limit("b2", Side.BUY, 99.99, 50)
    b.add_limit("b3", Side.BUY, 99.98, 300)
    b.add_limit("a1", Side.SELL, 100.01, 80)
    b.add_limit("a2", Side.SELL, 100.02, 200)
    return b


class OrderBookTests(unittest.TestCase):
    def test_quotes_and_analytics(self):
        b = book()
        self.assertEqual((b.best_bid(), b.best_ask(), b.spread(), b.mid()), (99.99, 100.01, 0.02, 100.0))
        self.assertEqual(b.depth(2), {"bids": [(99.99, 150.0), (99.98, 300.0)], "asks": [(100.01, 80.0), (100.02, 200.0)]})
        self.assertAlmostEqual(b.imbalance(1), (150 - 80) / 230)
        self.assertAlmostEqual(b.microprice(), (99.99 * 80 + 100.01 * 150) / 230)     # the heavier bid pulls it towards the ask
        self.assertGreater(b.microprice(), b.mid())
        self.assertEqual(len(b), 5)
        empty = OrderBook()
        self.assertEqual((empty.best_bid(), empty.mid(), empty.spread(), empty.microprice(), empty.imbalance()), (None, None, None, None, 0.0))

    def test_price_then_time_priority(self):
        b = book()
        trades = b.add_market("m1", Side.SELL, 120)
        self.assertEqual([(t.maker_id, t.quantity, t.price) for t in trades], [("b1", 100.0, 99.99), ("b2", 20.0, 99.99)])   # first in, first filled
        self.assertEqual((b.remaining("b2"), b.remaining("b1"), "b1" in b), (30.0, 0.0, False))
        trades = b.add_market("m2", Side.SELL, 100)
        self.assertEqual([(t.maker_id, t.price) for t in trades], [("b2", 99.99), ("b3", 99.98)])    # then the next price level
        self.assertEqual(b.last_price, 99.98)
        self.assertEqual(trades[0].aggressor, Side.SELL)

    def test_marketable_limit_trades_at_the_resting_price_and_rests_the_remainder(self):
        b = book()
        trades = b.add_limit("x", Side.BUY, 100.05, 100)                               # willing to pay 100.05 ...
        self.assertEqual([(t.price, t.quantity) for t in trades], [(100.01, 80.0), (100.02, 20.0)])   # ... gets the better resting prices
        b2 = book()
        b2.add_limit("y", Side.BUY, 100.01, 200)                                       # only 80 available at that price
        self.assertEqual((b2.remaining("y"), b2.best_bid(), b2.best_ask()), (120.0, 100.01, 100.02))

    def test_time_in_force(self):
        b = book()
        self.assertEqual(b.add_limit("fok", Side.BUY, 100.01, 100, tif=TimeInForce.FOK), [])          # cannot fill fully: does nothing
        self.assertEqual(b.remaining("a1"), 80.0)
        self.assertEqual(sum(t.quantity for t in b.add_limit("fok2", Side.BUY, 100.02, 100, tif=TimeInForce.FOK)), 100.0)
        ioc = b.add_limit("ioc", Side.BUY, 100.02, 500, tif=TimeInForce.IOC)
        self.assertEqual(sum(t.quantity for t in ioc), 180.0)                         # takes what is there ...
        self.assertNotIn("ioc", b)                                                    # ... and the rest is cancelled, not rested
        self.assertIsNone(b.best_ask())

    def test_post_only_and_market_order_leftover(self):
        b = book()
        with self.assertRaises(OrderError) as cm:
            b.add_limit("po", Side.BUY, 100.01, 10, post_only=True)
        self.assertEqual(cm.exception.code, "POST_ONLY_WOULD_CROSS")
        b.add_limit("po2", Side.BUY, 100.00, 10, post_only=True)                      # inside the spread without crossing is fine
        self.assertEqual(b.best_bid(), 100.00)
        trades = b.add_market("big", Side.BUY, 1_000)
        self.assertEqual(sum(t.quantity for t in trades), 280.0)                      # the book only had 280; the rest is dropped
        self.assertNotIn("big", b)

    def test_cancel_reduce_and_queue_position(self):
        b = book()
        self.assertEqual((b.queue_ahead("b1"), b.queue_ahead("b2")), (0.0, 100.0))
        b.reduce("b1", 40)
        self.assertEqual(b.queue_ahead("b2"), 40.0)                                   # reducing keeps the place in the queue
        self.assertEqual(b.cancel("b1"), 40.0)
        self.assertEqual(b.queue_ahead("b2"), 0.0)
        with self.assertRaises(OrderNotFoundError):
            b.cancel("b1")
        with self.assertRaises(OrderError):
            b.reduce("b2", 500)                                                       # growing would jump the queue: refused
        with self.assertRaises(OrderNotFoundError):
            b.queue_ahead("ghost")
        with self.assertRaises(OrderError):
            b.add_limit("b2", Side.BUY, 99.0, 1)                                      # duplicate id
        b.cancel("b2")
        self.assertEqual(b.best_bid(), 99.98)                                         # the emptied level is gone

    def test_self_trade_prevention(self):
        b = OrderBook(0.01)
        b.add_limit("mine", Side.SELL, 100.00, 50, owner="alice")
        b.add_limit("theirs", Side.SELL, 100.01, 50, owner="bob")
        self.assertEqual(b.add_limit("cross", Side.BUY, 100.01, 100, owner="alice"), [])   # would hit her own order first: nothing trades
        self.assertNotIn("cross", b)                                                  # and the incoming order is not left crossing the book
        self.assertEqual(len(b.add_market("m", Side.BUY, 50, owner="bob")), 1)
        loose = OrderBook(0.01, prevent_self_trade=False)
        loose.add_limit("s", Side.SELL, 100.0, 10, owner="alice")
        self.assertEqual(len(loose.add_limit("b", Side.BUY, 100.0, 10, owner="alice")), 1)

    def test_cost_to_trade_matches_what_actually_happens(self):
        b = book()
        est = b.cost_to_trade(Side.BUY, 150)
        self.assertAlmostEqual(est["average_price"], (80 * 100.01 + 70 * 100.02) / 150)
        self.assertEqual((est["filled"], est["unfilled"], est["levels"]), (150.0, 0.0, 2.0))
        self.assertAlmostEqual(est["slippage_bps"], (est["average_price"] - 100.0) / 100.0 * 1e4)
        self.assertEqual(len(b), 5)                                                   # an estimate changes nothing
        trades = b.add_market("m", Side.BUY, 150)
        self.assertAlmostEqual(sum(t.price * t.quantity for t in trades) / 150, est["average_price"])
        self.assertEqual(book().cost_to_trade(Side.SELL, 1_000)["unfilled"], 550.0)
        self.assertGreater(book().cost_to_trade(Side.SELL, 300)["slippage_bps"], 0)   # selling below the mid is also a cost

    def test_validation_and_tick_rounding(self):
        b = OrderBook(0.05)
        b.add_limit("o", Side.BUY, 10.02, 1)
        self.assertEqual(b.best_bid(), 10.0)                                          # snapped to the tick grid
        for call in (lambda: b.add_limit("z", Side.BUY, -1, 1), lambda: b.add_limit("z", Side.BUY, 1, 0),
                     lambda: b.add_market("z", Side.BUY, float("nan")), lambda: OrderBook(0)):
            with self.assertRaises(ValidationError):
                call()


class MarketSimulatorTests(unittest.TestCase):
    def test_same_seed_same_tape_and_a_sane_market(self):
        a, b = MarketSimulator(seed=7).run(3000), MarketSimulator(seed=7).run(3000)
        pd.testing.assert_frame_equal(a, b)
        self.assertFalse(MarketSimulator(seed=8).run(3000).equals(a))
        q = a.dropna(subset=["bid", "ask"])
        self.assertGreater(len(q), 2900)
        self.assertTrue((q["ask"] > q["bid"]).all())                                  # the book is never crossed
        self.assertTrue(q["time"].is_monotonic_increasing)
        self.assertGreater(q["bid"].max() - q["bid"].min(), 0.05)                     # the price actually moves

    def test_market_order_fills_at_the_quoted_prices_when_there_is_no_latency(self):
        sim = MarketSimulator(seed=1, latency=0.0)
        sim.run(2000)
        estimate = sim.book.cost_to_trade(Side.BUY, 200)
        oid = sim.submit(Side.BUY, 200)
        sim.run(1)
        self.assertEqual(sim.filled(oid), 200.0)
        self.assertAlmostEqual(sim.average_fill_price(oid), estimate["average_price"])

    def test_latency_delays_arrival(self):
        sim = MarketSimulator(seed=1, latency=2.0)
        sim.run(1000)
        oid = sim.submit(Side.BUY, 100)
        sent = sim.now
        while sim.now < sent + 2.0:
            self.assertEqual(sim.filled(oid), 0.0)                                    # not in the market yet
            sim.step()
        sim.run(5)
        self.assertEqual(sim.filled(oid), 100.0)

    def test_resting_order_waits_its_turn_in_the_queue(self):
        sim = MarketSimulator(seed=3, volatility=0.0)
        sim.run(500)
        bid = sim.book.best_bid()
        oid = sim.submit(Side.BUY, 100, price=bid)
        sim.run(1)
        if oid in sim.book:
            self.assertGreater(sim.book.queue_ahead(oid), 0)                          # joined behind the size already there
        sim.run(20_000)
        self.assertGreater(sim.filled(oid), 0)
        self.assertAlmostEqual(sim.average_fill_price(oid), bid)                      # a passive order is filled at its own price
        self.assertTrue(all(oid in (t.maker_id, t.taker_id) for t in sim.fills))

    def test_cancel_and_late_cancel(self):
        sim = MarketSimulator(seed=2, latency=0.5)
        sim.run(500)
        far = sim.submit(Side.BUY, 100, price=round(sim.book.best_bid() - 5.0, 2))    # far from the market: will not fill
        sim.run(100)
        self.assertIn(far, sim.book)
        sim.cancel(far)
        sim.run(100)
        self.assertNotIn(far, sim.book)
        self.assertEqual(sim.filled(far), 0.0)
        taker = sim.submit(Side.BUY, 50)
        sim.run(100)
        sim.cancel(taker)                                                             # already filled: the cancel arrives too late
        sim.run(100)
        self.assertEqual(sim.rejections[-1][0], taker)
        with self.assertRaises(OrderNotFoundError):
            sim.cancel("not-mine")
        with self.assertRaises(ValidationError):
            sim.submit(Side.BUY, 0)

    def test_parameter_validation(self):
        for kwargs in ({"event_rate": 0}, {"limit_share": 0.7, "market_share": 0.4}, {"latency": -1}, {"mean_distance": 0.5}, {"mid": 0}):
            with self.assertRaises(ValidationError, msg=str(kwargs)):
                MarketSimulator(**kwargs)
        with self.assertRaises(ValidationError):
            MarketSimulator().run(0)

    def test_tape_becomes_bars_the_backtester_accepts(self):
        sim = MarketSimulator(seed=5)
        quotes = sim.run(20_000)
        bars = ticks_to_bars(quotes, seconds=30)
        self.assertGreater(len(bars), 8)
        self.assertTrue(((bars["high"] >= bars[["open", "close"]].max(axis=1)) & (bars["low"] <= bars[["open", "close"]].min(axis=1))).all())
        self.assertEqual(int(bars["volume"].sum()), len(sim.book.trades))             # every trade is counted in exactly one bar
        self.assertEqual(bars.index[0], pd.Timestamp("2024-01-02 09:30:00"))
        self.assertEqual(len(sim.trade_tape()), len(sim.book.trades))
        md = MarketData()
        md.add("SIM", bars)                                                           # passes the platform's OHLCV validation
        self.assertEqual(len(md.frame("SIM")), len(bars))
        last = ticks_to_bars(quotes, 30, price="last")
        self.assertTrue((last["low"] <= last["high"]).all())
        for call in (lambda: ticks_to_bars(quotes, 0), lambda: ticks_to_bars(quotes, 30, price="vwap"),
                     lambda: ticks_to_bars(quotes.drop(columns="bid"), 30)):
            with self.assertRaises(ValidationError):
                call()


if __name__ == "__main__":
    unittest.main()
