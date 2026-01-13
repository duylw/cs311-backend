class PromptService:
    def __init__(self):
        self.prompts = {
            "retrieval_query_evaluate": "prompts/retrieval_query_evaluate.txt",
            "retrieval_generate_answer": "prompts/retrieval_generate_answer.txt",
            "retrieval_query_decompose": "prompts/retrieval_query_decompose.txt",
            "retrieval_query_enhance": "prompts/retrieval_query_enhance.txt",
        }
    
    def get_prompt(self, prompt_name: str) -> str:
        """Retrieve prompt template by name"""

        # Reads the entire content of a file
        with open(self.prompts[prompt_name], "r", encoding="utf-8") as file:
            prompt = file.read()

        return prompt

prompt_service = PromptService()