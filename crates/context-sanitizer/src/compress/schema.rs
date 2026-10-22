//! OpenAI JSON tool schema minification (`FR-6`).
//!
//! Recursively strips prose descriptions, documentation annotations, and redundant metadata
//! while strictly preserving tool names, parameter property keys, data types, enum definitions,
//! and required constraint arrays ($\ge 30\%$ token reduction with 100% schema validity).

use serde_json::Value;

/// Minifies an OpenAI `"tools": [...]` array in-place, removing verbose descriptions
/// and redundant schema metadata. Returns `(original_bytes, minified_bytes)`.
pub fn minify_tool_definitions(tools: &mut [Value]) -> (usize, usize) {
    let original_bytes: usize = tools
        .iter()
        .map(|t| serde_json::to_vec(t).map(|v| v.len()).unwrap_or(0))
        .sum();

    for tool in tools.iter_mut() {
        minify_value(tool);
    }

    let minified_bytes: usize = tools
        .iter()
        .map(|t| serde_json::to_vec(t).map(|v| v.len()).unwrap_or(0))
        .sum();

    (original_bytes, minified_bytes)
}

/// Recursively strips `"description"`, `"title"`, `"$schema"`, `"examples"`, and `"comment"` fields,
/// while strictly preserving parameter property definitions even if named `"title"` or `"description"`.
pub fn minify_value(val: &mut Value) {
    minify_value_inner(val, false);
}

fn minify_value_inner(val: &mut Value, is_properties_container: bool) {
    match val {
        Value::Object(map) => {
            if !is_properties_container {
                // Remove prose and non-essential documentation fields from schema nodes
                map.remove("description");
                map.remove("title");
                map.remove("$schema");
                map.remove("examples");
                map.remove("comment");
            }

            for (k, v) in map.iter_mut() {
                let is_props = k == "properties";
                minify_value_inner(v, is_props);
            }
        }
        Value::Array(arr) => {
            for item in arr {
                minify_value_inner(item, false);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
}

/// Computes the compression ratio achieved on the tools array.
#[must_use]
pub fn schema_reduction_ratio(original_bytes: usize, minified_bytes: usize) -> f32 {
    if original_bytes == 0 || minified_bytes >= original_bytes {
        0.0
    } else {
        ((original_bytes - minified_bytes) as f32) / (original_bytes as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn minifies_tool_schema_preserving_names_types_and_enums_over_30_percent() {
        let tool = json!({
            "type": "function",
            "function": {
                "name": "search_customer_records",
                "description": "Searches customer database by full name or email address and returns profile information with transaction history.",
                "parameters": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "title": "SearchCustomerParams",
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The customer search term such as email address or full name."
                        },
                        "department": {
                            "type": "string",
                            "enum": ["sales", "support", "billing", "engineering"],
                            "description": "Optional department filter for the search operation."
                        }
                    },
                    "required": ["query"]
                }
            }
        });

        let mut tools = vec![tool.clone()];
        let (orig, min) = minify_tool_definitions(&mut tools);
        let ratio = schema_reduction_ratio(orig, min);

        assert!(
            ratio >= 0.30,
            "Expected >= 30% reduction, achieved: {:.1}%",
            ratio * 100.0
        );

        let min_tool = &tools[0];
        // Critical structure preserved
        assert_eq!(
            min_tool["function"]["name"].as_str(),
            Some("search_customer_records")
        );
        assert_eq!(min_tool["function"]["parameters"]["type"], "object");
        assert_eq!(
            min_tool["function"]["parameters"]["properties"]["query"]["type"],
            "string"
        );
        assert_eq!(
            min_tool["function"]["parameters"]["properties"]["department"]["type"],
            "string"
        );
        assert!(min_tool["function"]["parameters"]["properties"]["department"]["enum"].is_array());
        assert_eq!(min_tool["function"]["parameters"]["required"][0], "query");

        // Prose stripped
        assert!(min_tool["function"].get("description").is_none());
        assert!(min_tool["function"]["parameters"]["properties"]["query"]
            .get("description")
            .is_none());
    }

    #[test]
    fn minifies_tool_schema_preserving_properties_named_title_or_description() {
        let mut tools = vec![json!({
            "type": "function",
            "function": {
                "name": "create_item",
                "description": "Create a new item in the backlog",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "The title of the item"
                        },
                        "description": {
                            "type": "string",
                            "description": "Detailed markdown description"
                        }
                    },
                    "required": ["title"]
                }
            }
        })];

        let (orig, min) = minify_tool_definitions(&mut tools);
        assert!(min < orig);
        let fn_obj = &tools[0]["function"];
        assert!(fn_obj.get("description").is_none());
        let props = &fn_obj["parameters"]["properties"];
        assert!(
            props.get("title").is_some(),
            "Property 'title' must be preserved!"
        );
        assert!(
            props.get("description").is_some(),
            "Property 'description' must be preserved!"
        );
        assert_eq!(props["title"]["type"], "string");
        assert_eq!(props["description"]["type"], "string");
        assert!(props["title"].get("description").is_none());
    }
}
