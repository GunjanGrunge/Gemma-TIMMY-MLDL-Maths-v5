import os
import json

def generate_with_gguf(prompt: str, gguf_path: str) -> str:
    """
    Mock inference function. 
    In the real environment, you use llama-cpp-python here:
    
    from llama_cpp import Llama
    llm = Llama(model_path=gguf_path, n_ctx=2048)
    response = llm(prompt, max_tokens=256)
    return response['choices'][0]['text']
    """
    
    # Simulating what the GGUF model returns for different prompts
    if "math" in prompt.lower() or "square root" in prompt.lower():
        # The V6.1 model fails to route general math, returning a failure JSON
        return '{"status": "no_v6_route", "explanation": "General mathematics not in consultant domain."}'
    else:
        # For actual machine learning/V6 questions, it correctly coordinates tools
        return '{"status": "success", "tool": "pca_calculator", "inputs": {"matrix": "data"}}'

def v61_hybrid_router_wrapper(question: str, gguf_path: str) -> dict:
    """
    This is the Helper Code to handle the GGUF output and implement the routing 
    fallback that was breaking our 'public_math_smoke' eval track.
    """
    print(f"\n--- Request: '{question}' ---")
    
    # 1. Format prompt for the model
    prompt = f"User: {question}\nConsultant:"
    
    # 2. Get raw execution from GGUF
    raw_output = generate_with_gguf(prompt, gguf_path)
    print(f"[Raw GGUF Output]: {raw_output}")
    
    # 3. Parse and apply the fallback logic
    try:
        data = json.loads(raw_output)
        
        # FIX: The critical fallback!
        # If the consultant throws 'no_v6_route', we explicitly catch it and route
        # it to the standard math pipeline, instead of fatally failing.
        if data.get("status") == "no_v6_route":
            print(">> [HELPER ACTION] Caught 'no_v6_route'! Rerouting to standard math node...")
            return {
                "final_route": "standard_math_solver",
                "v61_response": None,
                "note": "Question was elegantly redirected."
            }
            
        print(">> [HELPER ACTION] V6 domain confirmed. Executing tool plan.")
        return {
            "final_route": "v61_consultant",
            "v61_response": data,
            "note": "Processed successfully by GGUF model."
        }
        
    except json.JSONDecodeError:
        print(">> [HELPER ACTION] Invalid JSON out, falling back to math solver.")
        return {
            "final_route": "standard_math_solver",
            "v61_response": None,
            "note": "JSON Error Fallback"
        }

if __name__ == "__main__":
    GGUF_FILE = "outputs/v61/models/gemma_timmy_martha_v61_consultant_merged-unsloth.Q4_K_M.gguf"
    
    # Test 1: Standard Math (This previously scored 0% due to no fallback handling)
    test1 = "What is the square root of 144?"
    res1 = v61_hybrid_router_wrapper(test1, GGUF_FILE)
    print(f"Result: {json.dumps(res1, indent=2)}")
    
    # Test 2: V6 Domain Math
    test2 = "Explain the PCA algorithm matrix operations."
    res2 = v61_hybrid_router_wrapper(test2, GGUF_FILE)
    print(f"Result: {json.dumps(res2, indent=2)}")
