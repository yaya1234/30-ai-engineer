import pytest
import pandas as pd

def test_age_group():
    from analysis_report import age_group  # 如果函数在 analysis_report 中定义
    assert age_group(10) == 'Child'
    assert age_group(30) == 'Adult'
    assert age_group(70) == 'Senior'