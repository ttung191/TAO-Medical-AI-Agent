def _get_input(self, prompt, kwargs):
        p = prompt if prompt else (kwargs.get('prompt') or kwargs.get('content') or "Analyze")
        s = kwargs.get('system_instruction') or kwargs.get('system_prompt')
        # ✅ Khớp chính xác với model của Orchestrator
        m = kwargs.get('model') or "models/gemini-2.5-flash" 
        return str(p), s, m
