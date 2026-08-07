"""The console and the chat box beside the gating surface."""
import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.gate_console import evaluate


@pytest.fixture
def frame():
    return pd.DataFrame({"area": [10.0, 20.0, 30.0, 40.0],
                         "intensity": [1.0, 2.0, 3.0, 4.0]})


def test_a_column_can_be_named_directly(frame):
    """`area.mean()` is what people type; `df['area'].mean()` is what they
    write down afterwards."""
    assert evaluate("area.mean()", frame) == "25.0"


def test_the_frame_is_also_in_scope(frame):
    assert evaluate("df['area'].max()", frame) == "40.0"


def test_a_boolean_answer_is_counted(frame):
    """The commonest question while gating is "how many objects satisfy
    this", and a column of True/False is not an answer to it."""
    assert evaluate("area > 15", frame) == "3 of 4 objects"


def test_a_frame_answer_is_summarised_not_dumped(frame):
    assert evaluate("df[df.area > 15]", frame) == "3 rows × 2 columns"


def test_a_typo_comes_back_as_text(frame):
    """This is a question box; a typo is a normal thing to do in one."""
    answer = evaluate("area.men()", frame)
    assert "AttributeError" in answer
    assert "men" in answer


def test_an_empty_expression_answers_nothing(frame):
    assert evaluate("   ", frame) == ""


def test_with_no_table_it_says_so():
    assert evaluate("area.mean()", None) == "no table loaded"


def test_dangerous_builtins_are_not_in_scope(frame):
    """`__import__` and `open` in a box the user types into is a way to lose
    a dataset by typo, and no question this box exists for needs them."""
    assert "Error" in evaluate("__import__('os').listdir('.')", frame)
    assert "Error" in evaluate("open('/etc/passwd')", frame)


def test_a_statement_is_not_an_expression(frame):
    """Rebinding df would let the console and the plot disagree."""
    assert "Error" in evaluate("df = 1", frame)


def test_numpy_and_pandas_are_available(frame):
    assert evaluate("np.median(area)", frame) == "25.0"


# ---------------------------------------------------------------------------
# The widget
# ---------------------------------------------------------------------------

def test_the_transcript_records_both_halves(qtbot, frame):
    from spacr.qt.widgets.gate_console import GateConsole

    console = GateConsole()
    qtbot.addWidget(console)
    console.set_frame(frame)
    console.run("area.mean()")

    transcript = console.transcript()
    assert "area.mean()" in transcript and "25.0" in transcript


def test_the_chat_box_says_when_there_is_no_assistant(qtbot, frame):
    """A chat box that silently ignores you is worse than one that is
    honestly unavailable."""
    from spacr.qt.widgets.gate_console import GateConsole

    console = GateConsole()
    qtbot.addWidget(console)
    console.set_frame(frame)
    answer = console.ask("which population is which?")
    assert "no assistant is configured" in answer


def test_a_responder_answers_the_chat_box(qtbot, frame):
    from spacr.qt.widgets.gate_console import GateConsole

    console = GateConsole()
    qtbot.addWidget(console)
    console.set_responder(lambda q: f"you asked: {q}")
    assert console.ask("hello") == "you asked: hello"
    assert "you asked: hello" in console.transcript()


def test_a_broken_responder_does_not_take_the_screen_down(qtbot):
    from spacr.qt.widgets.gate_console import GateConsole

    def explode(_question):
        raise RuntimeError("no network")

    console = GateConsole()
    qtbot.addWidget(console)
    console.set_responder(explode)
    assert "could not answer" in console.ask("hello")


def test_the_question_is_emitted_for_an_async_host(qtbot):
    from spacr.qt.widgets.gate_console import GateConsole

    console = GateConsole()
    qtbot.addWidget(console)
    seen = []
    console.asked.connect(seen.append)
    console.ask("what is this cluster?")
    assert seen == ["what is this cluster?"]

    console.reply("probably debris")
    assert "probably debris" in console.transcript()


def test_the_screen_hands_its_table_to_the_console(qtbot, frame):
    from spacr.qt.linked_selection import LinkedSelection
    from spacr.qt.screens.gate_editor import GateEditorScreen

    screen = GateEditorScreen(link=LinkedSelection(), threaded=False)
    qtbot.addWidget(screen)
    screen.set_frame(frame)
    assert screen.console.run("len(df)") == "4"
