"""Simia code, copied into this recipe.

Source: microsoft/Simia-Agent-Training @ ea77d9d13362cc56cd1e0f33dc4cea4508aea28f (MIT, Copyright (c) 2025 Microsoft; see LICENSE-SIMIA).
  - fix_arguments_format, should_delete_conversation, process_conversation:
        Simia_SFT/Tau2/fix_arguments.py (unchanged)
  - parse_gpt_response: Simia_SFT/Tau2/utils/conversation_generator.py (method -> function, otherwise unchanged)
  - build_sample_text: Simia_SFT/Tau2/utils/data_loader.py (takes our canonical trace via ShareGPT)
  - format_reference_conversations: Simia-RL/.../simulated_General/env.py (method -> function)
The generation and environment prompts are in prompts.py (SIMIA_GEN, SIMIA_ENV).
"""
# ruff: noqa
import json
import re
from typing import Any, Dict, List

def fix_arguments_format(func_call_str: str) -> tuple[bool, str, str]:
    """
    Fix arguments format errors.
    Supports two formats:
    1. Pure JSON: {"name": "xxx", "arguments": {...}}
    2. Think + JSON: <think>xxx</think>\n{"name": "xxx", "arguments": {...}}
    Returns (success, fixed_content, message)
    """
    try:
        # Check for <think> tag
        think_content = ""
        json_part = func_call_str.strip()
        
        if "<think>" in func_call_str and "</think>" in func_call_str:
            think_match = re.search(r'<think>(.*?)</think>\s*(.*)', func_call_str, re.DOTALL)
            if think_match:
                think_content = f"<think>{think_match.group(1)}</think>\n"
                json_part = think_match.group(2).strip()
        
        func_call = json.loads(json_part)
        if 'arguments' in func_call:
            arguments = func_call['arguments']
            
            if isinstance(arguments, str):
                try:
                    parsed_args = json.loads(arguments)
                    func_call['arguments'] = parsed_args
                    fixed_json = json.dumps(func_call, ensure_ascii=False)
                    fixed_str = think_content + fixed_json
                    return True, fixed_str, "Fixed: string to dict"
                except json.JSONDecodeError:
                    if arguments.strip():
                        cleaned_args = arguments.strip()
                        if cleaned_args.startswith('{') and cleaned_args.endswith('}'):
                            try:
                                fixed_args = cleaned_args.replace("'", '"')
                                parsed_args = json.loads(fixed_args)
                                func_call['arguments'] = parsed_args
                                fixed_json = json.dumps(func_call, ensure_ascii=False)
                                fixed_str = think_content + fixed_json
                                return True, fixed_str, "Fixed: quotes"
                            except:
                                pass
                        func_call['arguments'] = {}
                        fixed_json = json.dumps(func_call, ensure_ascii=False)
                        fixed_str = think_content + fixed_json
                        return True, fixed_str, "Warning: empty dict"
                    else:
                        func_call['arguments'] = {}
                        fixed_json = json.dumps(func_call, ensure_ascii=False)
                        fixed_str = think_content + fixed_json
                        return True, fixed_str, "Fixed: empty to dict"
            elif isinstance(arguments, dict):
                final_str = think_content + json.dumps(func_call, ensure_ascii=False) if think_content else func_call_str
                return True, final_str, "No fix needed"
            else:
                func_call['arguments'] = {}
                fixed_json = json.dumps(func_call, ensure_ascii=False)
                fixed_str = think_content + fixed_json
                return True, fixed_str, f"Fixed: {type(arguments).__name__} to dict"
        else:
            func_call['arguments'] = {}
            fixed_json = json.dumps(func_call, ensure_ascii=False)
            fixed_str = think_content + fixed_json
            return True, fixed_str, "Fixed: added arguments"
            
    except json.JSONDecodeError as e:
        return False, func_call_str, f"Invalid JSON format: {str(e)}"
    except Exception as e:
        return False, func_call_str, f"Fix error: {str(e)}"


def should_delete_conversation(conversation: List[Dict[str, Any]]) -> bool:
    """Check if conversation should be deleted (contains '<tool_' or '\n[{')"""
    for turn in conversation:
        value = turn.get('value', '')
        if '<tool_' in value or '\n[{' in value:
            return True
    return False


def process_conversation(conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process conversation and fix function call arguments"""
    fixed_conversation = []
    
    for turn in conversation:
        if turn.get('from') == 'function_call':
            func_call_str = turn.get('value', '')
            is_fixed, fixed_str, _ = fix_arguments_format(func_call_str)
            
            if is_fixed:
                fixed_turn = turn.copy()
                fixed_turn['value'] = fixed_str
                fixed_conversation.append(fixed_turn)
            else:
                fixed_conversation.append(turn)
        else:
            fixed_conversation.append(turn)
    
    return fixed_conversation


def parse_gpt_response(response_text: str) -> Dict[str, Any]:
    """Parse GPT response and convert to ShareGPT format"""

    
    lines = response_text.strip().split('\n')
    conversations = []
    
    current_role = None
    current_content = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('HUMAN:') or line.startswith('H:'):
            if current_role and current_content:
                conversations.append({
                    "from": current_role,
                    "value": '\n'.join(current_content).strip()
                })
            current_role = "human"
            if line.startswith('HUMAN:'):
                current_content = [line[6:].strip()]  
            elif line.startswith('H:'):
                current_content = [line[2:].strip()] 
        elif line.startswith('ASSISTANT:') or line.startswith('A:'):
            if current_role and current_content:
                conversations.append({
                    "from": current_role,
                    "value": '\n'.join(current_content).strip()
                })
            current_role = "gpt" 
            if line.startswith('ASSISTANT:'):
                current_content = [line[10:].strip()]  
            elif line.startswith('A:'):
                current_content = [line[2:].strip()]  
        elif line.startswith('FUNCTION_CALL:'):
            if current_role and current_content:
                conversations.append({
                    "from": current_role,
                    "value": '\n'.join(current_content).strip()
                })
            current_role = "function_call"
            current_content = [line[14:].strip()]  
        elif line.startswith('OBSERVATION:'):
            if current_role and current_content:
                conversations.append({
                    "from": current_role,
                    "value": '\n'.join(current_content).strip()
                })
            current_role = "observation"
            current_content = [line[12:].strip()]  
        elif line and current_role:
            current_content.append(line)
    

    if current_role and current_content:
        conversations.append({
            "from": current_role,
            "value": '\n'.join(current_content).strip()
        })

    
    return {
        "conversations": conversations
    }


def build_sample_text(sample: Dict[str, Any]) -> str:
    """Build sample text for reference, including system and conversations (ShareGPT sample)."""
    text_parts = []

    system_content = sample.get('system', '')
    if system_content:
        text_parts.append(f"SYSTEM: {system_content}")

    conversations = sample.get('conversations', [])
    for turn in conversations:
        role = turn.get('from', '')
        content = turn.get('value', '')
        if role == 'human':
            text_parts.append(f"HUMAN: {content}")
        elif role == 'gpt':
            text_parts.append(f"ASSISTANT: {content}")
        elif role == 'function_call':
            text_parts.append(f"FUNCTION_CALL: {content}")
        elif role == 'observation':
            text_parts.append(f"OBSERVATION: {content}")

    return '\n\n'.join(text_parts)


def format_reference_conversations(reference_conversations: List[Dict[str, Any]]) -> str:
    """Format reference conversations as text"""
    if not reference_conversations:
        return "No reference conversation"

    lines = []
    for msg in reference_conversations:
        role = msg.get("from", "unknown")
        content = msg.get("value", "").strip()
        if content:
            if role == "human":
                lines.append(f"User/Tool response: {content}")
            elif role == "gpt" or role == "assistant":
                lines.append(f"Assistant: {content}")
            else:
                lines.append(f"{role}: {content}")

    return "\n\n".join(lines) if lines else "No reference conversation"
