from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from templates import template_processor # Assuming template_processor is accessible
from utils import ContentParser, DataAccessor # Assuming these are accessible
from models import TemplateValidationRequest, TemplateValidationResponse # Assuming these are accessible

router = APIRouter()

@router.get("/debug/node_content")
async def debug_node_content(node_content: str):
    """Debug endpoint to test content parsing functionality."""
    result = {
        "original_content": node_content,
        "analysis": {},
        "parsed_data": {}
    }
    
    # Analyze using ContentParser
    parser = ContentParser()
    numbered_items = parser.parse_numbered_list(node_content)
    json_data = parser.try_parse_json(node_content)
    table_data = parser.extract_table(node_content)
    
    # Build analysis
    result["analysis"] = {
        "has_numbered_list": bool(numbered_items),
        "numbered_items_count": len(numbered_items) if numbered_items else 0,
        "has_json": json_data is not None,
        "has_table": table_data is not None
    }
    
    # Add parsed data
    result["parsed_data"] = {
        "numbered_items": numbered_items,
        "json_data": json_data,
        "table_data": table_data
    }
    
    return result

@router.post("/debug/process_template")
async def debug_process_template(request: dict):
    """Debug endpoint to test template processing directly."""
    if "prompt" not in request or "context_data" not in request:
        raise HTTPException(status_code=400, detail="Request must include 'prompt' and 'context_data' fields")
    
    prompt = request["prompt"]
    context_data = request["context_data"]
    
    print(f"Debug process template request:")
    print(f"Prompt: {prompt}")
    print(f"Context data: {context_data}")
    
    # Process the template using our unified processor
    processed_prompt, processed_node_values = template_processor.process_node_references(
        prompt, context_data
    )
    
    # Return detailed results for debugging
    return {
        "original_prompt": prompt,
        "context_data": context_data,
        "processed_prompt": processed_prompt,
        "processed_node_values": processed_node_values,
        "validation": {
            "is_valid": len(template_processor.validate_node_references(prompt, context_data.keys())[1]) == 0,
            "missing_nodes": template_processor.validate_node_references(prompt, context_data.keys())[1],
            "found_nodes": template_processor.validate_node_references(prompt, context_data.keys())[2],
        }
    }

@router.post("/validate_template", response_model=TemplateValidationResponse)
async def validate_template_api(request: TemplateValidationRequest):
    """Validates that all node references in a template exist in the available nodes."""
    is_valid, missing_nodes, found_nodes = template_processor.validate_node_references(
        request.prompt_text, set(request.available_nodes)
    )
    
    warnings = []
    if not is_valid:
        for node in missing_nodes:
            warnings.append(f"Node reference '{node}' not found in available nodes.")
    
    return TemplateValidationResponse(
        is_valid=is_valid,
        missing_nodes=missing_nodes,
        found_nodes=found_nodes,
        warnings=warnings
    )

@router.post("/debug/test_reference")
async def debug_test_reference(request: dict):
    """Test endpoint for reference extraction."""
    if "content" not in request or "reference" not in request:
        raise HTTPException(status_code=400, detail="Request must include 'content' and 'reference' fields")
    
    content = request["content"]
    reference = request["reference"]
    
    # Create fake context with Node1 containing the content
    fake_context = {"Node1": content}
    data_accessor = DataAccessor(fake_context)
    
    # Try to parse the reference
    result = {
        "original_content": content,
        "reference": reference,
        "parsed_result": None,
        "details": {}
    }
    
    # Check if it's an item reference
    import re
    item_ref_pattern = r'\{([^:\}\[]+)(?:\[(\d+)\]|\:item\((\d+)\))\}'
    match = re.search(item_ref_pattern, reference)
    
    if match:
        node_name = match.group(1)
        item_num_str = match.group(2) or match.group(3)
        
        result["details"] = {
            "node_name": node_name,
            "item_num": item_num_str,
            "is_valid_node": data_accessor.has_node(node_name)
        }
        
        try:
            item_num = int(item_num_str)
            result["details"]["valid_item_num"] = True
            
            # Get the specific item
            item_content = data_accessor.get_item(node_name, item_num)
            result["parsed_result"] = item_content
            
        except ValueError:
            result["details"]["valid_item_num"] = False
    else:
        result["details"]["is_reference_pattern"] = False
    
    return result 