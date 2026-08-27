from egg.benchmark import main


def test_cpu_benchmark_smoke(capsys):
    main(
        [
            "--device",
            "cpu",
            "--hidden-size",
            "16",
            "--layers",
            "1",
            "--population",
            "2",
            "--sequence-length",
            "2",
            "--warmup",
            "0",
            "--iterations",
            "1",
            "--no-compile",
        ]
    )
    output = capsys.readouterr().out
    assert "throughput=" in output
    assert "population=2" in output
