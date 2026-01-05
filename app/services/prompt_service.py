class PromptService:
    def __init__(self):
        self.prompts = {
            "retrieval_query_translation": "prompts/retrieval_query_translation.txt",
            "generate_answer": "prompts/generate_answer.txt",
        }
    
    def get_prompt(self, prompt_name: str) -> str:
        """Retrieve prompt template by name"""

        # Reads the entire content of a file
        with open(self.prompts[prompt_name], "r", encoding="utf-8") as file:
            prompt = file.read()
        print(prompt)


        return prompt

prompt_service = PromptService()