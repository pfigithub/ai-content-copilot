from app.task_registry import (
    get_task_definition,
)


def test_summarize_task():

    task = get_task_definition(
        "summarize"
    )

    assert task.name == "summarize"
    assert "خلاصه" in task.instruction
    assert task.grounding_instruction != ""
    assert "summary" in task.output_instruction
    assert task.importance_instruction != ""
    assert (
        task.result_model.__name__
        == "SummaryResult"
    )


def test_analyze_task():

    task = get_task_definition(
        "analyze"
    )

    assert task.name == "analyze"
    assert "تحلیل" in task.instruction
    assert task.grounding_instruction != ""
    assert "analysis" in task.output_instruction
    assert task.importance_instruction != ""
    assert (
        task.result_model.__name__
        == "AnalysisResult"
    )


def test_ideas_task():

    task = get_task_definition(
        "ideas"
    )

    assert task.name == "ideas"
    assert "ایده" in task.instruction
    assert task.grounding_instruction != ""
    assert "ideas" in task.output_instruction
    assert task.importance_instruction == ""
    assert (
        task.result_model.__name__
        == "IdeasResult"
    )


def test_default_task():

    task = get_task_definition(
        None
    )

    assert task.name == "default"
    assert "عمومی" in task.instruction
    assert task.grounding_instruction != ""
    assert "summary" in task.output_instruction
    assert task.importance_instruction != ""
    assert (
        task.result_model.__name__
        == "SummaryResult"
    )