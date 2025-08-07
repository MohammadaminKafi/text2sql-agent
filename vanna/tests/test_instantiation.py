from vanna.mock import MockEmbedding, MockLLM, MockVectorDB


def test_mock_components():
    embedder = MockEmbedding()
    llm = MockLLM()
    vectordb = MockVectorDB()

    e = embedder.generate_embedding("data")
    assert isinstance(e, list)

    r = llm.submit_prompt(
        [
            llm.system_message("test"),
            llm.user_message("hi"),
        ]
    )
    assert isinstance(r, str)

    ddl_id = vectordb.add_ddl("CREATE TABLE t(id INT)")
    assert isinstance(ddl_id, str)
