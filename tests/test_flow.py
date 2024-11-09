from nodeflow.builtin import IF


def test_if_expression():
    true_case  = 5
    false_case = 0

    assert IF.compute(True, true_case, false_case)  == true_case
    assert IF.compute(False, true_case, false_case) == false_case
