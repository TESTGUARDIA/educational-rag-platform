import subprocess
import sys

import pytest

from projet_ia.cli import main


def test_aide_principale(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    assert "verifier" in capsys.readouterr().out


@pytest.mark.parametrize("args", [["verifier", "--help"], ["calibrer", "--help"],
                                  ["detection", "preparer", "--help"],
                                  ["detection", "genetique", "--help"],
                                  ["detection", "entrainer", "--help"],
                                  ["detection", "predire", "--help"]])
def test_aide_des_sous_commandes(args):
    r = subprocess.run([sys.executable, "-m", "projet_ia", *args],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_commande_inconnue():
    with pytest.raises(SystemExit) as e:
        main(["nimportequoi"])
    assert e.value.code != 0


def test_langue_invalide():
    with pytest.raises(SystemExit) as e:
        main(["detection", "preparer", "--langue", "klingon"])
    assert e.value.code != 0
