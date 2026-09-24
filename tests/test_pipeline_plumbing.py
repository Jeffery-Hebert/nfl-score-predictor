"""
The plumbing between pipeline stages: provenance sidecars, the composite, the
parallel backtest runner, the weekly command, the pull manifest and kickoff
times. None of it computes a forecast, and every piece can still corrupt one --
a stale backtest read as current, a composite averaged over different games on
different rows, a failed member hidden behind an exit code of 0, a forecast run
on an injury report that was never final. All on synthetic inputs, no data/.

Run: pytest tests/test_pipeline_plumbing.py -v
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import schedule, weekly
from src.ingest import manifest
from src.models import composite, registry, run_all
from src.validate import backtest_io


def results_frame(n=6, offset=0.0):
    return pd.DataFrame(
        {
            "game_id": [f"2026_01_A{i}_H{i}" for i in range(n)],
            "season": 2026,
            "week": 1,
            "home_score": np.arange(n, dtype=float) + 20,
            "away_score": np.arange(n, dtype=float) + 17,
            "home_pred": np.arange(n, dtype=float) + 21 + offset,
            "away_pred": np.arange(n, dtype=float) + 18 + offset,
        }
    )


# ---------------------------------------------------------------- sidecars --


class TestProvenanceSidecar:
    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        monkeypatch.setattr(backtest_io, "PRED_DIR", tmp_path / "processed")
        table = tmp_path / "model_table.parquet"
        pd.DataFrame(
            {
                "game_id": ["g1", "g2", "g3"],
                "home_score": [24.0, 17.0, np.nan],  # g3 not yet played
                "feature": [0.1, 0.2, 0.3],
            }
        ).to_parquet(table, index=False)
        backtest_io.save_predictions(results_frame(), "m", inputs=[table])
        return table

    def rewrite(self, table, gid, col, value):
        df = pd.read_parquet(table)
        df.loc[df["game_id"] == gid, col] = value
        df.to_parquet(table, index=False)

    def test_a_fresh_backtest_is_current_and_records_its_metrics(self, env):
        assert backtest_io.stale_inputs("m") == []
        meta = backtest_io.load_meta("m")
        assert meta["n_games"] == 6 and meta["seasons"] == [2026]
        assert meta["metrics"]["home_rmse"] == pytest.approx(1.0)
        assert meta["inputs"][0]["played_sha256"]

    def test_a_change_to_an_unplayed_row_is_not_staleness(self, env):
        self.rewrite(env, "g3", "feature", 9.9)
        assert backtest_io.stale_inputs("m") == []

    def test_a_change_to_a_played_row_is(self, env):
        self.rewrite(env, "g1", "feature", 9.9)
        (problem,) = backtest_io.stale_inputs("m")
        assert "played games changed" in problem

    def test_a_game_being_played_is_staleness(self, env):
        self.rewrite(env, "g3", "home_score", 30.0)
        assert backtest_io.stale_inputs("m")

    def test_row_order_alone_is_not_staleness(self, env):
        df = pd.read_parquet(env)
        df.iloc[::-1].to_parquet(env, index=False)
        assert backtest_io.stale_inputs("m") == []

    def test_an_edited_prediction_file_is_caught(self, env):
        path = backtest_io.predictions_path("m")
        results_frame(offset=1.0).to_parquet(path, index=False)
        assert any("modified after" in p for p in backtest_io.stale_inputs("m"))

    def test_a_vanished_input_or_sidecar_is_staleness(self, env):
        env.unlink()
        assert any("no longer exists" in p for p in backtest_io.stale_inputs("m"))
        assert "no provenance sidecar" in backtest_io.stale_inputs("never_run")[0]

    def test_a_file_without_scores_falls_back_to_its_whole_hash(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(backtest_io, "PRED_DIR", tmp_path / "processed")
        seqs = tmp_path / "sequences.npz"
        np.savez(seqs, a=np.arange(3))
        backtest_io.save_predictions(results_frame(), "m", inputs=[seqs])
        assert backtest_io.load_meta("m")["inputs"][0]["played_sha256"] is None
        np.savez(seqs, a=np.arange(4))
        assert any("changed since" in p for p in backtest_io.stale_inputs("m"))

    def test_old_sidecars_are_upgraded_only_where_provably_unchanged(
        self, env, tmp_path
    ):
        other = tmp_path / "other.parquet"
        pd.DataFrame({"game_id": ["g1"], "home_score": [1.0]}).to_parquet(other)
        backtest_io.save_predictions(results_frame(), "m", inputs=[env, other])
        meta = backtest_io.load_meta("m")
        for rec in meta["inputs"]:  # as written before fingerprints existed
            rec.pop("played_sha256")
        backtest_io.meta_path("m").write_text(json.dumps(meta))
        pd.DataFrame({"game_id": ["g1"], "home_score": [2.0]}).to_parquet(other)

        assert backtest_io.add_fingerprints("m") == [str(env)]
        recs = {r["path"]: r for r in backtest_io.load_meta("m")["inputs"]}
        assert recs[str(env)]["played_sha256"]
        assert not recs[str(other)].get("played_sha256")
        # The changed file is still reported: the upgrade never launders it.
        assert backtest_io.stale_inputs("m") == [
            f"{other} changed since this backtest ran"
        ]


# --------------------------------------------------------------- composite --


class TestCompositeCombine:
    def test_it_is_the_mean_over_games_every_member_predicted(self):
        a = results_frame(n=6)
        b = results_frame(n=4, offset=2.0)  # b skipped the last two games
        out = composite.combine({"a": a, "b": b})
        assert len(out) == 4
        assert np.allclose(out["home_pred"], a["home_pred"][:4] + 1.0)
        assert np.allclose(out["away_pred"], a["away_pred"][:4] + 1.0)

    def test_a_duplicated_game_is_refused(self):
        a = results_frame()
        dup = pd.concat([a, a.iloc[[0]]])
        with pytest.raises(pd.errors.MergeError):
            composite.combine({"a": a, "b": dup})

    def test_no_members_is_refused(self):
        with pytest.raises(ValueError):
            composite.combine({})

    def test_members_scored_against_different_results_are_refused(self):
        # A member backtest run on an older table: dropping the game quietly
        # would make the composite's game set depend on which member is stale.
        a = results_frame()
        b = a.copy()
        b.loc[0, "home_score"] = 99.0
        with pytest.raises(ValueError, match="disagree"):
            composite.combine({"a": a, "b": b})


# -------------------------------------------------------------- the runner --


class TestRunAll:
    @pytest.fixture
    def fake(self, monkeypatch):
        """run_one replaced by a recorder; nothing is actually executed."""
        calls, outcome = [], {}

        def run_one(spec, threads):
            calls.append(spec.name)
            return {
                "name": spec.name,
                "ok": outcome.get(spec.name, True),
                "seconds": 0.0,
                "log": f"logs/{spec.name}.log",
            }

        monkeypatch.setattr(run_all, "run_one", run_one)
        monkeypatch.setattr(run_all, "load_meta", lambda name: None)
        monkeypatch.setattr(run_all, "missing_requirement", lambda spec: None)
        return SimpleNamespace(calls=calls, outcome=outcome, mp=monkeypatch)

    def test_everything_runs_after_its_dependencies(self, fake):
        assert run_all.main(["--jobs", "3"]) == 0
        assert sorted(fake.calls) == sorted(registry.names())
        for name in fake.calls:
            spec = registry.get(name)
            for dep in run_all.dependencies(spec):
                assert fake.calls.index(dep) < fake.calls.index(name), (dep, name)

    def test_a_failed_member_fails_its_dependents_and_the_run(self, fake, capsys):
        fake.outcome["gp"] = False
        assert run_all.main(["--only", "linear", "gp", "poisson", "composite"]) == 1
        assert "composite" not in fake.calls
        assert "a dependency failed" in capsys.readouterr().out

    def test_a_missing_optional_dependency_skips_without_failing(self, fake, capsys):
        fake.mp.setattr(
            run_all,
            "missing_requirement",
            lambda spec: "torch not installed" if spec.name == "rnn" else None,
        )
        assert run_all.main(["--only", "rnn", "linear"]) == 0
        assert fake.calls == ["linear"]
        assert "skipped" in capsys.readouterr().out

    def test_bad_arguments_stop_before_anything_runs(self, fake):
        with pytest.raises(SystemExit):
            run_all.main(["--jobs", "0"])
        with pytest.raises(SystemExit, match="unknown model"):
            run_all.main(["--only", "posson"])
        assert fake.calls == []

    def test_list_runs_nothing(self, fake, capsys):
        assert run_all.main(["--list"]) == 0
        assert fake.calls == []
        assert "composite" in capsys.readouterr().out


# ------------------------------------------------------ the weekly command --


class TestWeekly:
    @pytest.fixture
    def commands(self, monkeypatch):
        ran, fail_on = [], {}

        def fake_run(cmd, *a, **k):
            ran.append(cmd)
            module = cmd[2] if len(cmd) > 2 else ""
            return SimpleNamespace(returncode=fail_on.get(module, 0))

        monkeypatch.setattr(weekly.subprocess, "run", fake_run)
        return SimpleNamespace(ran=ran, fail_on=fail_on)

    @staticmethod
    def modules(ran):
        return [cmd[2] for cmd in ran]

    def test_the_normal_run_pulls_builds_checks_then_predicts(self, commands):
        assert weekly.main([]) == 0
        assert self.modules(commands.ran) == [
            "src.ingest.pull_all",
            "src.features.build_all",
            "pytest",
            "src.predict.predict_week",
        ]
        assert weekly.DATA_CHECKS in commands.ran[2]
        assert "--allow-unsettled-injuries" not in commands.ran[-1]

    def test_flags_reach_the_predictor(self, commands):
        weekly.main(
            [
                "--skip-pull",
                "--allow-unsettled-injuries",
                "--season",
                "2026",
                "--week",
                "3",
            ]
        )
        assert "src.ingest.pull_all" not in self.modules(commands.ran)
        predict = commands.ran[-1]
        assert predict[-5:] == [
            "--allow-unsettled-injuries",
            "--season",
            "2026",
            "--week",
            "3",
        ]

    def test_a_failed_step_stops_the_run_with_its_exit_code(self, commands):
        commands.fail_on["pytest"] = 3
        with pytest.raises(SystemExit) as e:
            weekly.main(["--skip-pull"])
        assert e.value.code == 3
        assert "src.predict.predict_week" not in self.modules(commands.ran)

    def test_backtest_adds_the_runner_and_the_report(self, commands):
        weekly.main(["--skip-pull", "--backtest"])
        mods = self.modules(commands.ran)
        assert mods.index("src.models.run_all") < mods.index(
            "src.validate.model_report"
        )
        assert mods.index("src.validate.model_report") < mods.index(
            "src.predict.predict_week"
        )


# ------------------------------------------------------------ the manifest --


class TestManifest:
    @pytest.fixture
    def pulled(self, tmp_path, monkeypatch):
        monkeypatch.setattr(manifest, "MANIFEST", tmp_path / "_pull_manifest.json")
        path = tmp_path / "injuries.parquet"
        df = pd.DataFrame({"season": [2019, 2025], "report_status": ["Out", "Out"]})
        df.to_parquet(path, index=False)
        # Asked for through 2026; 2026 had not been published yet.
        manifest.record_pull("injuries", path, df, seasons=list(range(2019, 2027)))
        return path

    def test_it_records_what_arrived_not_what_was_asked_for(self, pulled):
        rec = manifest.read()["datasets"]["injuries"]
        assert rec["seasons"] == [2019, 2025]
        assert rec["rows"] == 2

    def test_the_pull_time_is_trusted_while_the_file_is_unchanged(self, pulled):
        t = manifest.pulled_at("injuries")
        assert t is not None
        assert abs(pd.Timestamp.now(tz="UTC") - t) < pd.Timedelta(minutes=5)

    def test_a_replaced_file_voids_the_recorded_time(self, pulled):
        pd.DataFrame({"season": [2026]}).to_parquet(pulled, index=False)
        assert manifest.pulled_at("injuries") is None

    def test_unknown_or_missing_datasets_have_no_time(self, pulled):
        assert manifest.pulled_at("snap_counts") is None
        pulled.unlink()
        assert manifest.pulled_at("injuries") is None

    def test_validation_results_do_not_clobber_pulls(self, pulled):
        manifest.record_validation({"errors": [], "warnings": ["w"]})
        m = manifest.read()
        assert "injuries" in m["datasets"] and m["validation"]["warnings"] == ["w"]


# ---------------------------------------------------------- kickoff times --


class TestKickoff:
    def frame(self, *rows):
        return pd.DataFrame(rows, columns=["game_id", "gameday", "gametime"])

    def test_eastern_time_converts_across_daylight_saving(self):
        s = self.frame(
            ("tnf", "2026-09-24", "20:15"),  # EDT, UTC-4: crosses midnight
            ("dec", "2026-12-06", "13:00"),  # EST, UTC-5
            ("london", "2026-10-11", "09:30"),
        )
        got = schedule.kickoff_utc(s)
        assert list(got) == [
            pd.Timestamp("2026-09-25 00:15", tz="UTC"),
            pd.Timestamp("2026-12-06 18:00", tz="UTC"),
            pd.Timestamp("2026-10-11 13:30", tz="UTC"),
        ]

    def test_a_missing_gametime_is_a_1pm_slot_and_garbage_is_nat(self):
        s = self.frame(("a", "2026-09-27", None), ("b", "not a date", "13:00"))
        got = schedule.kickoff_utc(s)
        assert got.iloc[0] == pd.Timestamp("2026-09-27 17:00", tz="UTC")
        assert pd.isna(got.iloc[1])

    def test_by_game_restricts_and_indexes(self):
        s = self.frame(("a", "2026-09-27", "13:00"), ("b", "2026-09-28", "20:15"))
        got = schedule.kickoff_by_game(s, ["b"])
        assert list(got.index) == ["b"]
        assert got["b"] == pd.Timestamp("2026-09-29 00:15", tz="UTC")


# ------------------------------------------------- feature-build freshness --


class TestInputFingerprint:
    CONFIG = (
        "# a comment\n"
        "training:\n  recency_half_life_weeks: 17\n"
        "live:\n  models: [poisson, gp, linear]\n"
    )

    def fingerprint(self, tmp_path, text):
        from src.features.build_all import input_fingerprint

        path = tmp_path / "config.yaml"
        path.write_text(text)
        return input_fingerprint(str(path))

    def test_config_edits_outside_training_do_not_stale_the_features(self, tmp_path):
        before = self.fingerprint(tmp_path, self.CONFIG)
        edited = self.CONFIG.replace("# a comment", "# another comment").replace(
            "[poisson, gp, linear]", "[poisson, linear]"
        )
        assert self.fingerprint(tmp_path, edited) == before

    def test_a_training_change_does(self, tmp_path):
        before = self.fingerprint(tmp_path, self.CONFIG)
        changed = self.CONFIG.replace(": 17", ": 26")
        assert self.fingerprint(tmp_path, changed) != before

    def test_any_other_input_is_its_content_hash(self, tmp_path):
        from src.features.build_all import input_fingerprint
        from src.provenance import sha256

        path = tmp_path / "table.parquet"
        pd.DataFrame({"a": [1]}).to_parquet(path)
        assert input_fingerprint(str(path)) == sha256(path)
