
import sys
import pytest
from io import StringIO
from contextlib import contextmanager
from gprMax.input_cmd_funcs import *

@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err

def test_rx():
    with captured_output() as (out, err):
        rx(0, 0, 0)
    assert out.getvalue().strip() == '#rx: 0 0 0'

def test_rx2():
    with captured_output() as (out, err):
        rx(0, 1, 2, 'id')
    assert out.getvalue().strip() == '#rx: 0 1 2 id'

def test_rx3():
    with captured_output() as (out, err):
        rx(2, 1, 0, 'idd', ['Ex'])
    assert out.getvalue().strip() == '#rx: 2 1 0 idd Ex'

def test_rx4():
    with captured_output() as (out, err):
        rx(2, 1, 0, 'id', ['Ex', 'Ez'])
    assert out.getvalue().strip() == '#rx: 2 1 0 id ExEz'

def test_rx_rotate_exception():
    with pytest.raises(ValueError):
        rx(2, 1, 0, 'id', ['Ex', 'Ez'], polarisation='x', rotate90origin=(1, 1))

def test_rx_rotate_success():
    with captured_output() as (out, err):
        rx(2, 1, 0, 'id', ['Ex', 'Ez'], polarisation='x', rotate90origin=(1, 1), dxdy=(0, 0))
    assert out.getvalue().strip() == '#rx: 1 2 0 id ExEz'

def test_rx_rotate_success2():
    with captured_output() as (out, err):
        rx(2, 1, 0, 'id', ['Ex', 'Ez'], polarisation='y', rotate90origin=(1, 1), dxdy=(0, 0))
    assert out.getvalue().strip() == '#rx: 1 2 0 id ExEz'

def test_src_steps():
    with captured_output() as (out, err):
        src_steps()
    assert out.getvalue().strip() == '#src_steps: 0 0 0'

def test_src_steps2():
    with captured_output() as (out, err):
        src_steps(42, 43, 44.2)
    assert out.getvalue().strip() == '#src_steps: 42 43 44.2'

def test_rx_steps():
    with captured_output() as (out, err):
        rx_steps()
    assert out.getvalue().strip() == '#rx_steps: 0 0 0'

def test_rx_steps2():
    with captured_output() as (out, err):
        rx_steps(42, 43, 44.2)
    assert out.getvalue().strip() == '#rx_steps: 42 43 44.2'
