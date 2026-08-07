import path
import pytest

IS_LAPLAMA_SERVER = path.Path("/fefs/aswg/data/real").exists()
from lst_ulities.helper import find_lst_data_path


@pytest.mark.skipif(not IS_LAPLAMA_SERVER, reason="This test need to run at the LAPLAMA server")
def test_find_dl1_file_old():
    date = 20250323
    run_number = 20463
    dl1_path = find_lst_data_path(date, run_number)
    assert dl1_path == "/fefs/aswg/data/real/DL1/20250323/v0.10/tailcut1005/dl1_LST-1.Run20463.h5"


@pytest.mark.skipif(not IS_LAPLAMA_SERVER, reason="This test need to run at the LAPLAMA server")
def test_find_dl1_file_new():
    date = 20250621
    run_number = 20758
    dl1_path = find_lst_data_path(date, run_number)
    assert dl1_path == "/fefs/onsite/data/lst-pipe/LSTN-01/DL1/20250621/v0.11/tailcut84/dl1_LST-1.Run20758.h5"
