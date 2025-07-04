import json
import pandas as pd
from datetime import datetime
import re # For cleaning text
import os # For checking file existence
from pprint import pprint
import concurrent.futures

# --- Configuration (for data loading, not models) ---
ES_DATA_FILE = 'es_data.json'

# --- Precomputed Call Trends ---

CALLS_PER_DAY = {
    "2025-01-27": 432,
    "2025-01-28": 2084,
    "2025-01-29": 3038,
    "2025-01-30": 2886,
    "2025-06-11": 517,
    "2025-06-12": 519,
}

CALLS_PER_DEVICE = {
    "BAVERN": {"incoming": 0, "outgoing": 4526},
    "DESKTOP-R5CIDDA": {"incoming": 0, "outgoing": 3086},
    "Unknown Device": {"incoming": 695, "outgoing": 613},
    "70:85:C2:E6:66:79": {"incoming": 0, "outgoing": 677},
    "70:85:C2:12:69:4F": {"incoming": 0, "outgoing": 676},
    "A8:A1:59:28:2D:45": {"incoming": 0, "outgoing": 288},
    "70:85:C2:19:9F:E3": {"incoming": 0, "outgoing": 84},
    "DESKTOP-19RCLU4": {"incoming": 0, "outgoing": 16},
}

CALLS_PER_HOUR = {
    "00": 530, "01": 533, "02": 545, "03": 532, "04": 345, "05": 426, "06": 285, "07": 475,
    "08": 373, "09": 488, "10": 633, "11": 455, "12": 427, "13": 505, "14": 347, "15": 354,
    "16": 445, "17": 290, "18": 266, "19": 266, "20": 266, "21": 267, "22": 440, "23": 527
}

TOTAL_DURATION_PER_DEVICE = {
    "BAVERN": 1630000,
    "DESKTOP-R5CIDDA": 1050000,
    "A8:A1:59:28:2D:45": 670000,
    "70:85:C2:19:9F:E3": 670000,
    "DESKTOP-19RCLU4": 360000,
    "70:85:C2:12:69:4F": 290000,
    "70:85:C2:E6:66:79": 270000,
    "Unknown": 10000,
}

# --- Data Loading and Preparation ---

def load_es_data(file_path):
    """Loads Elasticsearch response from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['hits']['hits']
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        print(f"Creating a dummy '{file_path}' for testing purposes.")
        # Create a dummy ES data file for initial testing if it doesn't exist
        dummy_data = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "comid": "comm_123",
                            "@timestamp": "2025-06-24T10:00:00.000Z",
                            "@timestampend": "2025-06-24T10:05:00.000Z",
                            "fields": {
                                "system": {"name": "Oversight", "systemType": "CallRecording"},
                                "deviceInfo": {"deviceID": "DEV001", "audioInterface": "Handset 1"},
                                "communicationRecord": {
                                    "comUUID": "uuid_abc", "startTime": "2025-06-24T10:00:00.000Z",
                                    "direction": "outgoing", "universalID": "uid1", "csID": "csid1",
                                    "activeDuration": 300, "comType": "VoiceCall", "endTime": "2025-06-24T10:05:00.000Z",
                                    "platform": {"name": "btt"}
                                },
                                "participants": [
                                    {"involvement": "source", "address": "192.168.1.10", "member": {"name": "Alice"}},
                                    {"involvement": "destination", "address": "192.168.1.11", "member": {"name": "Bob"}}
                                ],
                                "comFiles": [{"fileType": "wav", "audioInterface": "Handset 1"}],
                                "linkedData": [{"systemType": "CTI", "isMissing": True}, {"systemType": "CRM", "isMissing": False}]
                            }
                        }
                    },
                    {
                        "_source": {
                            "comid": "comm_124",
                            "@timestamp": "2025-06-24T11:00:00.000Z",
                            "@timestampend": "2025-06-24T11:10:00.000Z",
                            "fields": {
                                "system": {"name": "RecorderX", "systemType": "CallRecording"},
                                "deviceInfo": {"deviceID": "DEV002", "audioInterface": "DDI"},
                                "communicationRecord": {
                                    "comUUID": "uuid_def", "startTime": "2025-06-24T11:00:00.000Z",
                                    "direction": "incoming", "universalID": "uid2", "csID": "csid2",
                                    "activeDuration": 600, "comType": "VideoCall", "endTime": "2025-06-24T11:10:00.000Z",
                                    "platform": {"name": "Teams"}
                                },
                                "participants": [
                                    {"involvement": "source", "address": "192.168.1.12", "member": {"name": "Charlie"}},
                                    {"involvement": "destination", "address": "192.168.1.13", "member": {"name": "David"}}
                                ],
                                "comFiles": [{"fileType": "mp4", "audioInterface": "Webcam"}],
                                "linkedData": [{"systemType": "CRM", "isMissing": True}, {"systemType": "SAP", "isMissing": False}]
                            }
                        }
                    },
                     {
                        "_source": {
                            "comid": "comm_125",
                            "@timestamp": "2025-06-23T09:00:00.000Z",
                            "@timestampend": "2025-06-23T09:02:00.000Z",
                            "fields": {
                                "system": {"name": "Oversight", "systemType": "CallRecording"},
                                "deviceInfo": {"deviceID": "DEV003", "audioInterface": "Handset 2"},
                                "communicationRecord": {
                                    "comUUID": "uuid_ghi", "startTime": "2025-06-23T09:00:00.000Z",
                                    "direction": "outgoing", "universalID": "uid3", "csID": "csid3",
                                    "activeDuration": 120, "comType": "VoiceCall", "endTime": "2025-06-23T09:02:00.000Z",
                                    "platform": {"name": "btt"}
                                },
                                "participants": [
                                    {"involvement": "source", "address": "192.168.1.14", "member": {"name": "Eve"}},
                                    {"involvement": "destination", "address": "192.168.1.15", "member": {"name": "Frank"}}
                                ],
                                "comFiles": [],
                                "linkedData": []
                            }
                        }
                    },
                    {
                        "_source": {
                            "comid": "comm_126",
                            "@timestamp": "2025-06-24T12:00:00.000Z",
                            "@timestampend": "2025-06-24T12:01:00.000Z",
                            "fields": {
                                "system": {"name": "Oversight", "systemType": "CallRecording"},
                                "deviceInfo": {"deviceID": "DEV004", "audioInterface": "Handset 1"},
                                "communicationRecord": {
                                    "comUUID": "uuid_jkl", "startTime": "2025-06-24T12:00:00.000Z",
                                    "direction": "incoming", "universalID": "uid4", "csID": "csid4",
                                    "activeDuration": 60, "comType": "VoiceCall", "endTime": "2025-06-24T12:01:00.000Z",
                                    "platform": {"name": "btt"}
                                },
                                "participants": [
                                    {"involvement": "source", "address": "192.168.1.16", "member": {"name": "Grace"}},
                                    {"involvement": "destination", "address": "192.168.1.17", "member": {"name": "Heidi"}}
                                ],
                                "comFiles": [{"fileType": "wav", "audioInterface": "Handset 1"}],
                                "linkedData": []
                            }
                        }
                    },
                    {
                        "_source": {
                            "comid": "comm_127",
                            "@timestamp": "2025-06-25T08:30:00.000Z",
                            "@timestampend": "2025-06-25T08:35:00.000Z",
                            "fields": {
                                "system": {"name": "RecorderX", "systemType": "CallRecording"},
                                "deviceInfo": {"deviceID": "DEV005", "audioInterface": "Microphone"},
                                "communicationRecord": {
                                    "comUUID": "uuid_mno", "startTime": "2025-06-25T08:30:00.000Z",
                                    "direction": "outgoing", "universalID": "uid5", "csID": "csid5",
                                    "activeDuration": 300, "comType": "ConferenceCall", "endTime": "2025-06-25T08:35:00.000Z",
                                    "platform": {"name": "Teams"}
                                },
                                "participants": [
                                    {"involvement": "source", "address": "192.168.1.18", "member": {"name": "Ivan"}},
                                    {"involvement": "destination", "address": "192.168.1.19", "member": {"name": "Judy"}}
                                ],
                                "comFiles": [{"fileType": "wav", "audioInterface": "Conference Mic"}],
                                "linkedData": [{"systemType": "Video", "isMissing": True}]
                            }
                        }
                    }
                ]
            }
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(dummy_data, f, indent=2)
        return dummy_data['hits']['hits']
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'. Please check file format.")
        return []

def flatten_record(record_source):
    """Flatten a nested Elasticsearch _source record into a flat dictionary, supporting all fields in the data."""
    flattened = {}

    # Top-level fields (including @timestamp, comid, etc.)
    for key, value in record_source.items():
        if key == 'fields':
            continue  # We'll handle below
        if isinstance(value, dict):
            for subk, subv in value.items():
                flattened[f'{key}_{subk}'] = subv
        else:
            flattened[key.replace('@', '')] = value  # Remove @ from timestamp keys

    # Flatten all fields in 'fields' (may include nested dicts/lists)
    fields = record_source.get('fields', {})
    for fkey, fval in fields.items():
        if isinstance(fval, dict):
            for subk, subv in fval.items():
                flattened[f'{fkey}_{subk}'] = subv
        elif isinstance(fval, list):
            # For list of dicts (e.g., comFiles, participants, linkedData)
            if all(isinstance(item, dict) for item in fval):
                # For each subfield, collect all values as comma-separated strings
                subfield_values = {}
                for item in fval:
                    for subk, subv in item.items():
                        # If nested dict (e.g., checksum, member), flatten further
                        if isinstance(subv, dict):
                            for subsubk, subsubv in subv.items():
                                keyname = f'{fkey}_{subk}_{subsubk}'
                                subfield_values.setdefault(keyname, []).append(str(subsubv) if subsubv is not None else "")
                        else:
                            keyname = f'{fkey}_{subk}'
                            subfield_values.setdefault(keyname, []).append(str(subv) if subv is not None else "")
                # Add as comma-separated strings
                for subfield, values in subfield_values.items():
                    flattened[subfield] = ", ".join(values)
                # Also add count
                flattened[f'{fkey}_count'] = len(fval)
            else:
                # List of primitives
                flattened[f'{fkey}'] = ", ".join(str(x) for x in fval)
                flattened[f'{fkey}_count'] = len(fval)
        else:
            flattened[fkey] = fval

    return flattened

def prepare_documents_for_chroma_and_df(es_records):
    """Parses timestamps and prepares documents for ChromaDB and a Pandas DataFrame."""
    df_rows = []

    for i, record in enumerate(es_records):
        record_source = record['_source']
        flattened_data = flatten_record(record_source)

        df_rows.append(flattened_data) # Keep original data for DataFrame flexibility

    df = pd.DataFrame(df_rows)
    # Convert timestamp columns to datetime objects for proper filtering/grouping
    # Safely convert specific known datetime columns.
    datetime_fields = [
        'timestamp', 'timestampend',
        'communicationRecord_startTime', 'communicationRecord_endTime'
    ]
    for col in datetime_fields:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)

    # --- NEW: Calculate duration from start and end times ---
    # Ensure both startTime and endTime are not NaT before calculating duration
    df['communicationRecord_calculatedDuration'] = (df['communicationRecord_endTime'] - df['communicationRecord_startTime']).dt.total_seconds().fillna(0)
    # --- END NEW ---

    # Re-generate documents and metadatas with the new calculatedDuration
    documents = []
    metadatas = []
    ids = []
    for i, row in df.iterrows():
        # Create a text content for embedding (semantic search)
        content_parts = []
        if pd.notnull(row.get('communicationRecord_comType')):
            content_parts.append(f"{row['communicationRecord_comType']} event")
        if pd.notnull(row.get('communicationRecord_direction')):
            content_parts.append(f"direction {row['communicationRecord_direction']}")
        if pd.notnull(row.get('participant_source_name')):
            content_parts.append(f"from {row['participant_source_name']}")
        if pd.notnull(row.get('participant_destination_name')):
            content_parts.append(f"to {row['participant_destination_name']}")
        if pd.notnull(row.get('communicationRecord_startTime')):
            content_parts.append(f"starting at {row['communicationRecord_startTime'].isoformat()}")
        if pd.notnull(row.get('deviceInfo_audioInterface')):
            content_parts.append(f"via {row['deviceInfo_audioInterface']}")
        if pd.notnull(row.get('communicationRecord_platform_name')):
            content_parts.append(f"on platform {row['communicationRecord_platform_name']}")
        # Use the newly calculated duration field
        if pd.notnull(row.get('communicationRecord_calculatedDuration')):
             content_parts.append(f"duration {row['communicationRecord_calculatedDuration']} seconds")
        # Fallback to activeDuration if calculatedDuration is not available/relevant (though calculatedDuration should be more robust now)
        elif pd.notnull(row.get('communicationRecord_activeDuration')):
             content_parts.append(f"duration {row['communicationRecord_activeDuration']} seconds")


        document_content = ". ".join(content_parts)
        documents.append(document_content)

        # Replace None/NaT values in metadata with appropriate defaults for ChromaDB
        # Convert row to dictionary for metadata and ensure Timestamps are strings
        cleaned_metadata = {}
        for k, v in row.items():
            # Fix: handle lists/arrays robustly
            if isinstance(v, (list, tuple)):
                cleaned_metadata[k] = ", ".join(str(x) for x in v)
            elif hasattr(v, 'dtype') and hasattr(v, 'tolist'):
                # numpy array
                cleaned_metadata[k] = ", ".join(str(x) for x in v.tolist())
            elif pd.isna(v):
                cleaned_metadata[k] = "" # Replace NaN/NaT with empty string
            elif isinstance(v, pd.Timestamp):
                cleaned_metadata[k] = v.isoformat() # Convert Timestamp to ISO string
            else:
                cleaned_metadata[k] = v
        metadatas.append(cleaned_metadata)
        ids.append(f"doc_{i}")

    return documents, metadatas, ids, df

# --- Query Router / Intent Classifier ---

def simple_keyword_intent_classifier(query):
    """Simple rule-based fallback for intent classification."""
    query_lower = query.lower()
    aggregation_keywords = [
        "how many", "count", "total", "average", "sum", "trend", "trends", "list calls on", "calls by", "longest", "shortest", "list", "number of",
        "no of", "no ", "num of", "num "
    ]
    semantic_keywords = [
        "describe", "details", "information about", "tell me about", "related to", "specific call", "provide details on"
    ] # Added more semantic keywords

    if any(keyword in query_lower for keyword in aggregation_keywords):
        return {"query_type": "aggregation", "reason": "Keyword-based classification."}
    if any(keyword in query_lower for keyword in semantic_keywords):
        return {"query_type": "semantic", "reason": "Keyword-based classification."}

    # If no strong match, default to 'uncertain' to trigger both pipelines
    return {"query_type": "uncertain", "reason": "Default to uncertain due to ambiguous keywords."}

def classify_query_intent(query, llm_pipeline):
    """Classifies user query as 'semantic', 'aggregation', or 'uncertain' using LLM."""
    system_prompt = """You are an intelligent assistant that classifies user queries about call records.
    Determine if the query requires a 'semantic' search (e.g., 'find calls related to X', 'describe details', 'explain'), an 'aggregation' operation (e.g., 'count', 'sum', 'average', 'trends', 'how many', 'list', 'longest', 'shortest'), or if it's 'uncertain' (meaning it could potentially involve both or is ambiguous).

    Output only a JSON object with 'query_type' (semantic, aggregation, or uncertain) and an optional 'reason'). Ensure the output is a single, valid JSON object and nothing else. If you are unsure, default to 'uncertain'.
    """

    few_shot_examples = """
    User: How many calls were made yesterday?
    Output: {"query_type": "aggregation", "reason": "Requires counting calls based on date."}

    User: What is the average duration of calls from participant 5651?
    Output: {"query_type": "aggregation", "reason": "Requires calculating average duration filtered by participant."}

    User: Show me call trends for Handset 1 by daily counts.
    Output: {"query_type": "aggregation", "reason": "Requires time series aggregation with grouping."}

    User: List calls where the CTI data was missing.
    Output: {"query_type": "aggregation", "reason": "Requires listing records with a specific filter (missing data)."}

    User: How many calls were outgoing voice calls?
    Output: {"query_type": "aggregation", "reason": "Requires counting based on call type and direction."}

    User: Describe calls related to the 'Oversight' system.
    Output: {"query_type": "semantic", "reason": "Requires semantic understanding and description of calls."}

    User: What were the calls made from device 'DDI'?
    Output: {"query_type": "aggregation", "reason": "Requires listing records filtered by device."}

    User: Provide details on the call starting at 2025-06-12T11:24:16.000Z.
    Output: {"query_type": "semantic", "reason": "Requires semantic search for a specific record's details."}

    User: Is there any information about calls with duration more than 10 seconds?
    Output: {"query_type": "aggregation", "reason": "Requires listing or counting calls based on a numeric filter."}

    User: Show me calls on platform btt with audio interface Handset 2.
    Output: {"query_type": "aggregation", "reason": "Requires listing records with multiple filters."}

    User: How many calls had missing linked data of type 'video'?
    Output: {"query_type": "aggregation", "reason": "Requires counting records with specific missing linked data type."}

    User: List calls that used 'Handset 1' audio interface from comFiles.
    Output: {"query_type": "aggregation", "reason": "Requires listing records filtered by specific audio interface in comFiles."}

    User: Tell me about the general communication patterns.
    Output: {"query_type": "uncertain", "reason": "Could involve both aggregation (trends) and semantic (description) aspects."}

    User: Summarize everything about calls on platform Teams.
    Output: {"query_type": "uncertain", "reason": "Requires both aggregation (counts, durations) and semantic (key themes) summarization."}

    User: What is the average number of calls per day in 2025?
    Output: {
      "operation": "time_series_count",
      "filters": [
        {"field": "communicationRecord_startTime", "operator": "date_after", "value": "2025-01-01"},
        {"field": "communicationRecord_startTime", "operator": "date_before", "value": "2026-01-01"}
      ],
      "group_by": "communicationRecord_startTime",
      "group_by_interval": "daily",
      "target_field": null
    }

    User: What is the average number of calls per hour on 2025-06-12?
    Output: {
      "operation": "time_series_count",
      "filters": [
        {"field": "communicationRecord_startTime", "operator": "date_equals", "value": "2025-06-12"}
      ],
      "group_by": "communicationRecord_startTime",
      "group_by_interval": "hourly",
      "target_field": null
    }

    User: Show call trends per month in 2025.
    Output: {
      "operation": "time_series_count",
      "filters": [
        {"field": "communicationRecord_startTime", "operator": "date_after", "value": "2025-01-01"},
        {"field": "communicationRecord_startTime", "operator": "date_before", "value": "2026-01-01"}
      ],
      "group_by": "communicationRecord_startTime",
      "group_by_interval": "monthly",
      "target_field": null
    }
    """

    prompt_template = """<bos><start_of_turn>user
{system_prompt}

{few_shot_examples}

User: {user_query}
<end_of_turn><start_of_turn>model
"""
    prompt = prompt_template.format(
        system_prompt=system_prompt,
        few_shot_examples=few_shot_examples,
        user_query=query
    )

    try:
        response_raw = llm_pipeline(prompt, max_new_tokens=150, temperature=0.1)[0]['generated_text']
        clean_response = response_raw.split('<start_of_turn>model\n')[-1].strip()

        # Extract JSON from potential markdown block
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', clean_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else: # Fallback if no markdown block (Gemma sometimes omits it)
            json_match = re.search(r'\{.*\}', clean_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                print(f"Warning: Could not extract JSON from LLM response for intent classification: {clean_response}")
                return simple_keyword_intent_classifier(query) # Fallback to keyword classifier

        json_str = json_str.strip()
        # Post-processing for common Gemma JSON errors
        if json_str.endswith(',}'):
            json_str = json_str[:-2] + '}'
        if not json_str.endswith('}'):
            json_str += '}'
        json_str = json_str.replace('"null"', 'null') # Fix "null" as string to actual null

        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error for intent classification: {e} from response: {clean_response}")
        return simple_keyword_intent_classifier(query)
    except Exception as e:
        print(f"Error during LLM classification: {e}")
        return simple_keyword_intent_classifier(query)

# --- Aggregation Path Logic ---

SCHEMA_FIELDS = {
    "comid": {"type": "string", "description": "Unique communication ID."},
    "timestamp": {"type": "datetime", "description": "Overall record start timestamp."},
    "timestampend": {"type": "datetime", "description": "Overall record end timestamp."},
    "communicationRecord_startTime": {"type": "datetime", "description": "Start time of the specific communication record/call."},
    "communicationRecord_endTime": {"type": "datetime", "description": "End time of the specific communication record/call."},
    "communicationRecord_activeDuration": {"type": "numeric", "description": "Duration of the call in seconds (e.g., 60, 300). This field might be zero or null for some records. If you need a more reliable duration, use communicationRecord_calculatedDuration."},
    "communicationRecord_calculatedDuration": {"type": "numeric", "description": "Calculated duration in seconds based on endTime - startTime. Use this for more accurate duration analysis."}, # New field
    "communicationRecord_comType": {"type": "string", "description": "Type of communication, e.g., 'VoiceCall', 'Chat'."},
    "communicationRecord_direction": {"type": "string", "description": "Direction of the call, 'outgoing' or 'incoming'."},
    "communicationRecord_platform_name": {"type": "string", "description": "Name of the communication platform, e.g., 'btt', 'Teams'."},
    "deviceInfo_deviceID": {"type": "string", "description": "Unique identifier for the device."},
    "deviceInfo_channelID": {"type": "string", "description": "Channel identifier for the device."},
    "deviceInfo_pUserID": {"type": "string", "description": "Participant user ID from device info."},
    "deviceInfo_audioInterface": {"type": "string", "description": "Audio interface used, e.g., 'Handset 1', 'DDI', 'Microphone'."},
    "deviceInfo_buttonName": {"type": "string", "description": "Button name pressed on the device."},
    "participant_source_address": {"type": "string", "description": "Network address of the source participant."},
    "participant_source_name": {"type": "string", "description": "Name or ID of the source participant (e.g., '5651', 'John Doe')."},
    "participant_destination_name": {"type": "string", "description": "Name or ID of the destination participant."},
    "participant_destination_address": {"type": "string", "description": "Network address of the destination participant."},
    "system_name": {"type": "string", "description": "Name of the recording system, e.g., 'Oversight', 'RecorderX'."},
    "system_systemType": {"type": "string", "description": "Type of the recording system."},
    "comFiles_count": {"type": "numeric", "description": "Number of associated communication files."},
    "comFiles_types": {"type": "string", "description": "Comma-separated file types of communication files, e.g., 'wav, mp3', 'txt'."},
    "comFiles_audioInterfaces": {"type": "string", "description": "Comma-separated audio interfaces mentioned in communication files, e.g., 'Handset 1', 'Conference Mic'."},
    "linkedData_missing_types": {"type": "string", "description": "Comma-separated types of linked data that are explicitly marked as missing, e.g., 'CTI', 'CRM', 'video'."},
    "linkedData_present_types": {"type": "string", "description": "Comma-separated types of linked data that are present."},
}


def generate_aggregation_query(user_query, llm_pipeline, schema_fields):
    """
    Robust rule-based aggregation query generator. Handles synonyms, natural language dates, and flexible field-value phrasing.
    """
    import re
    from dateutil import parser as date_parser

    user_query_lower = user_query.lower()

    # Synonym and field mapping
    field_map = {
        # comFiles
        'filetype': 'comFiles_fileType',
        'file type': 'comFiles_fileType',
        'audiointerface': 'comFiles_audioInterface',
        'audio interface': 'comFiles_audioInterface',
        'platform': 'communicationRecord_platform_name',
        'platform name': 'communicationRecord_platform_name',
        # participants
        'participant address': 'participants_address',
        'participant name': 'participants_member_name',
        'participant id': 'participants_member_pUserID',
        'source address': 'participant_source_address',
        'destination address': 'participant_destination_address',
        # linkedData
        'missing linkeddata': 'linkedData_missing_types',
        'present linkeddata': 'linkedData_present_types',
        'linkeddata': 'linkedData_missing_types',
        'linked data': 'linkedData_missing_types',
        # deviceInfo
        'deviceid': 'deviceInfo_deviceID',
        'audiointerface': 'deviceInfo_audioInterface',
        'buttonname': 'deviceInfo_buttonName',
        # communicationRecord
        'direction': 'communicationRecord_direction',
        'comtype': 'communicationRecord_comType',
        'date': 'communicationRecord_startTime',
        'day': 'communicationRecord_startTime',
        'month': 'communicationRecord_startTime',
        'year': 'communicationRecord_startTime',
        'duration': 'communicationRecord_calculatedDuration',
    }

    # Helper: Try to parse a natural language date
    def try_parse_date(text):
        try:
            dt = date_parser.parse(text, fuzzy=True)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None

    # --- Trend/Time Series ---
    if re.search(r"(trend|trends|pattern|patterns|distribution|over time|evolution|progression|calls per day|calls per month|calls per hour)", user_query_lower):
        if "month" in user_query_lower:
            return {"operation": "count_per_month"}
        if "hour" in user_query_lower:
            return {"operation": "count_per_hour"}
        return {"operation": "count_per_day"}

    # --- Min/Max/Avg/Sum ---
    if re.search(r"(average|avg|mean)", user_query_lower):
        return {"operation": "average_duration"}
    if re.search(r"(minimum|min)", user_query_lower):
        return {"operation": "min_duration"}
    if re.search(r"(maximum|max)", user_query_lower):
        return {"operation": "max_duration"}
    if re.search(r"(sum|total) duration", user_query_lower):
        return {"operation": "sum_duration"}

    # --- Directional Count ---
    m = re.search(r"(incoming|outgoing|inbound|outbound) calls?(?:.*?(?:in|on|during)\s+(\d{4}))?", user_query_lower)
    if m:
        direction = m.group(1)
        direction = {"inbound": "incoming", "outbound": "outgoing"}.get(direction, direction)
        filters = [{"field": "communicationRecord_direction", "operator": "equals", "value": direction}]
        if m.group(2):
            filters.append({"field": "communicationRecord_startTime", "operator": "year_equals", "value": m.group(2)})
        return {"operation": "count", "filters": filters}

    # --- Platform Count ---
    m = re.search(r"(?:calls? (?:on|with|using) )([\w\-]+) platform", user_query_lower)
    if m:
        platform = m.group(1)
        return {"operation": "count", "filters": [{"field": "communicationRecord_platform_name", "operator": "equals", "value": platform}]}

    # --- Field-Value Count (generic, e.g. "calls with filetype wav") ---
    m = re.search(r"calls? (?:with|where|having) ([\w\s]+?)(?: is| equals| =|:| to)? ([\w@.\-]+)", user_query_lower)
    if m:
        field_raw = m.group(1).strip()
        value = m.group(2).strip()
        field_key = field_raw.replace(' ', '').lower()
        field = field_map.get(field_key)
        if not field:
            # Try direct match to DataFrame columns
            for col in schema_fields:
                if field_raw.replace(' ', '').lower() in col.replace('_', '').lower():
                    field = col
                    break
        if field:
            return {"operation": "count", "filters": [{"field": field, "operator": "contains", "value": value}]}
        else:
            return {"error": "Relevant data not found."}

    # --- Date-based Count (e.g. "calls made on June 24, 2025") ---
    m = re.search(r"calls? (?:made )?on ([a-zA-Z]+ \d{1,2}(?:st|nd|rd|th)?(?:,? \d{4})?)", user_query_lower)
    if m:
        date_str = try_parse_date(m.group(1))
        if date_str:
            return {"operation": "count", "filters": [{"field": "communicationRecord_startTime", "operator": "date_equals", "value": date_str}]}

    # --- Date-based Count (YYYY-MM-DD, YYYY-MM, YYYY) ---
    m = re.search(r"calls? (?:made )?on (\d{4}-\d{2}-\d{2})", user_query_lower)
    if m:
        return {"operation": "count", "filters": [{"field": "communicationRecord_startTime", "operator": "date_equals", "value": m.group(1)}]}
    m = re.search(r"calls? (?:made )?on (\d{4}-\d{2})", user_query_lower)
    if m:
        return {"operation": "count", "filters": [{"field": "communicationRecord_startTime", "operator": "month_equals", "value": m.group(1)}]}
    m = re.search(r"calls? (?:made )?in (\d{4}-\d{2})", user_query_lower)
    if m:
        return {"operation": "count", "filters": [{"field": "communicationRecord_startTime", "operator": "month_equals", "value": m.group(1)}]}
    m = re.search(r"calls? (?:made )?in (\d{4})", user_query_lower)
    if m:
        return {"operation": "count", "filters": [{"field": "communicationRecord_startTime", "operator": "year_equals", "value": m.group(1)}]}

    # --- Month name (natural language) + year, e.g. "in january 2025", "in jun 2025" ---
    m = re.search(r"in (jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|september|oct|october|nov|november|dec|december)(?:[ ,]+(\d{4}))?", user_query_lower)
    if m:
        month_map = {
            'jan': '01', 'january': '01', 'feb': '02', 'february': '02', 'mar': '03', 'march': '03',
            'apr': '04', 'april': '04', 'may': '05', 'jun': '06', 'june': '06', 'jul': '07', 'july': '07',
            'aug': '08', 'august': '08', 'sep': '09', 'september': '09', 'oct': '10', 'october': '10',
            'nov': '11', 'november': '11', 'dec': '12', 'december': '12'
        }
        month_str = m.group(1)
        year_str = m.group(2) if m.group(2) else None
        month_num = month_map[month_str]
        if year_str:
            ym = f"{year_str}-{month_num}"
            return {"operation": "count", "filters": [{"field": "communicationRecord_startTime", "operator": "month_equals", "value": ym}]}
        else:
            # Only month given, filter by month (any year)
            return {"operation": "count", "filters": [{"field": "communicationRecord_startTime", "operator": "month_only", "value": month_num}]}

    # --- Fallback: Try to match any field in the schema with a value ---
    for field, meta in schema_fields.items():
        if field in user_query_lower:
            m = re.search(rf"{field} (is|=|equals|to)? ([\w@.\-]+)", user_query_lower)
            if m:
                value = m.group(2).strip()
                return {"operation": "count", "filters": [{"field": field, "operator": "contains", "value": value}]}

    # --- Default: Just count all calls ---
    if re.search(r"(how many|total|count|number of|no of|no |num of|num |amount of) calls?", user_query_lower):
        return {"operation": "count"}

    return {"error": "Relevant data not found."}

def execute_aggregation_query(df, agg_query_json):
    """
    Efficiently executes the aggregation query described by agg_query_json on the DataFrame df.
    Uses vectorized boolean masking for all filters and robust groupby/aggregation logic.
    """
    if not agg_query_json:
        return {"error": "Invalid or unsupported aggregation query."}
    op = agg_query_json.get("operation")
    filters = agg_query_json.get("filters", [])
    mask = pd.Series([True] * len(df))
    for f in filters:
        field = f.get("field")
        optr = f.get("operator")
        value = f.get("value")
        if field not in df.columns:
            print(f"Warning: Filter field '{field}' not found in DataFrame. Skipping filter.")
            continue
        col = df[field]
        # Datetime filters
        if pd.api.types.is_datetime64_any_dtype(col):
            try:
                if optr == "date_equals":
                    value_dt = pd.to_datetime(value, utc=True) if isinstance(value, str) else value
                    mask &= (col.dt.date == value_dt.date())
                elif optr == "month_equals":
                    value_dt = pd.to_datetime(value, utc=True) if isinstance(value, str) else value
                    mask &= (col.dt.strftime('%Y-%m') == value_dt.strftime('%Y-%m'))
                elif optr == "month_only":
                    # value is a month number as string (e.g., '03'), do not parse as date
                    mask &= (col.dt.strftime('%m') == str(value).zfill(2))
                elif optr == "year_equals":
                    mask &= (col.dt.strftime('%Y') == str(value))
                elif optr == "date_before":
                    value_dt = pd.to_datetime(value, utc=True) if isinstance(value, str) else value
                    mask &= (col < value_dt)
                elif optr == "date_after":
                    value_dt = pd.to_datetime(value, utc=True) if isinstance(value, str) else value
                    mask &= (col > value_dt)
                else:
                    print(f"Warning: Unsupported datetime operator '{optr}' for field '{field}'.")
            except Exception as e:
                print(f"Error applying date filter for field '{field}' with value '{value}': {e}.")
        # Numeric filters
        elif pd.api.types.is_numeric_dtype(col):
            try:
                num_val = float(value)
                if optr == "greater_than":
                    mask &= (col > num_val)
                elif optr == "less_than":
                    mask &= (col < num_val)
                elif optr == "equals":
                    mask &= (col == num_val)
                elif optr == "not_equals":
                    mask &= (col != num_val)
                else:
                    print(f"Warning: Unsupported numeric operator '{optr}' for field '{field}'.")
            except Exception as e:
                print(f"Error casting value '{value}' to numeric for field '{field}': {e}.")
        # String filters
        else:
            col_str = col.fillna('').astype(str)
            val_str = str(value)
            if optr == "equals":
                mask &= (col_str.str.lower() == val_str.lower())
            elif optr == "not_equals":
                mask &= (col_str.str.lower() != val_str.lower())
            elif optr == "contains":
                mask &= (col_str.str.contains(val_str, case=False, na=False))
            elif optr == "not_contains":
                mask &= (~col_str.str.contains(val_str, case=False, na=False))
            elif optr == "starts_with":
                mask &= (col_str.str.startswith(val_str, na=False))
            elif optr == "ends_with":
                mask &= (col_str.str.endswith(val_str, na=False))
            elif optr == "is_null":
                mask &= (col.isnull())
            elif optr == "is_not_null":
                mask &= (col.notnull())
            else:
                print(f"Warning: Unknown operator '{optr}' for field '{field}'.")
    df_filtered = df[mask]
    # ...existing code for aggregation operations...
    if op == "count":
        return {"count": int(len(df_filtered))}
    if op == "count_month":
        return {"count": int(len(df_filtered))}
    if op == "count_year":
        return {"count": int(len(df_filtered))}
    if op == "count_per_day":
        if "communicationRecord_startTime" in df_filtered.columns:
            counts = df_filtered.groupby(df_filtered["communicationRecord_startTime"].dt.date).size().to_dict()
            return {"count_per_day": counts}
        else:
            return {"error": "No start time column for per-day aggregation."}
    if op == "count_per_month":
        if "communicationRecord_startTime" in df_filtered.columns:
            counts = df_filtered.groupby(df_filtered["communicationRecord_startTime"].dt.to_period('M')).size().to_dict()
            counts = {str(k): v for k, v in counts.items()}
            return {"count_per_month": counts}
        else:
            return {"error": "No start time column for per-month aggregation."}
    if op == "count_per_hour":
        if "communicationRecord_startTime" in df_filtered.columns:
            counts = df_filtered.groupby(df_filtered["communicationRecord_startTime"].dt.hour).size().to_dict()
            return {"count_per_hour": counts}
        else:
            return {"error": "No start time column for per-hour aggregation."}
    if op == "average_duration":
        if "communicationRecord_calculatedDuration" in df_filtered.columns:
            avg = df_filtered["communicationRecord_calculatedDuration"].mean()
            return {"average_duration": float(avg) if not pd.isna(avg) else 0.0}
        else:
            return {"error": "No duration column for average calculation."}
    if op == "sum_duration":
        if "communicationRecord_calculatedDuration" in df_filtered.columns:
            total = df_filtered["communicationRecord_calculatedDuration"].sum()
            return {"sum_duration": float(total) if not pd.isna(total) else 0.0}
        else:
            return {"error": "No duration column for sum calculation."}
    if op == "min_duration":
        if "communicationRecord_calculatedDuration" in df_filtered.columns:
            minval = df_filtered["communicationRecord_calculatedDuration"].min()
            return {"min_duration": float(minval) if not pd.isna(minval) else 0.0}
        else:
            return {"error": "No duration column for min calculation."}
    if op == "max_duration":
        if "communicationRecord_calculatedDuration" in df_filtered.columns:
            maxval = df_filtered["communicationRecord_calculatedDuration"].max()
            return {"max_duration": float(maxval) if not pd.isna(maxval) else 0.0}
        else:
            return {"error": "No duration column for max calculation."}
    return {"error": "Aggregation operation not implemented."}


def synthesize_aggregation_result(user_query, raw_result, llm_pipeline, agg_query=None):
    """
    Synthesizes a human-readable summary of the aggregation result using the LLM.
    For trend/aggregation queries, returns a summary string.
    """
    if not raw_result or (isinstance(raw_result, dict) and 'error' in raw_result):
        return f"No relevant data found for your query."
    # Human-readable summaries for common aggregations
    if 'count' in raw_result:
        return f"Total calls: {raw_result['count']}"
    if 'count_per_day' in raw_result:
        counts = raw_result['count_per_day']
        if not counts:
            return "No call data available for the requested days."
        summary = "Call trends per day:\n" + "\n".join([f"{k}: {v}" for k, v in counts.items()])
        return summary
    if 'count_per_month' in raw_result:
        counts = raw_result['count_per_month']
        if not counts:
            return "No call data available for the requested months."
        summary = "Call trends per month:\n" + "\n".join([f"{k}: {v}" for k, v in counts.items()])
        return summary
    if 'count_per_hour' in raw_result:
        counts = raw_result['count_per_hour']
        if not counts:
            return "No call data available for the requested hours."
        summary = "Call trends per hour:\n" + "\n".join([f"{k}: {v}" for k, v in counts.items()])
        return summary
    if 'average_duration' in raw_result:
        return f"Average call duration: {raw_result['average_duration']:.2f} seconds"
    if 'sum_duration' in raw_result:
        return f"Total call duration: {raw_result['sum_duration']:.2f} seconds"
    if 'min_duration' in raw_result:
        return f"Minimum call duration: {raw_result['min_duration']:.2f} seconds"
    if 'max_duration' in raw_result:
        return f"Maximum call duration: {raw_result['max_duration']:.2f} seconds"
    # Fallback
    return f"Aggregation result: {raw_result}"

def is_meaningless_query(query, llm_pipeline=None):
    """
    Returns True if the query is likely meaningless or nonsensical for the call records domain.
    If llm_pipeline is provided, uses the LLM to determine meaningfulness; if LLM fails, falls back to rule-based check.
    Always treat queries with aggregation keywords as meaningful.
    """
    query_lower = query.lower()
    # Always treat aggregation queries as meaningful
    aggregation_keywords = ["how many", "total", "count", "number of", "sum", "average", "min", "max", "trend"]
    if any(kw in query_lower for kw in aggregation_keywords):
        return False
    if llm_pipeline is not None:
        try:
            result = is_meaningful_query_llm(query, llm_pipeline)
            if result is not None:
                return not result
        except Exception as e:
            print(f"LLM error in is_meaningful_query_llm: {e}")
    # Fallback: rule-based check
    known_keywords = [
        "call", "calls", "duration", "participant", "device", "platform", "direction", "audio", "interface", "system", "trend", "average", "sum", "count", "total", "number", "list", "date", "day", "hour", "minute", "outgoing", "incoming", "voice", "video", "conference", "crm", "cti", "sap", "btt", "oversight", "recorderx", "handset", "ddi", "microphone"
    ]
    if not any(kw in query_lower for kw in known_keywords):
        return True
    unrelated_words = ["kannappa", "collections"]
    if any(word in query_lower for word in unrelated_words):
        return True
    return False

def is_meaningful_query_llm(query, llm_pipeline):
    """
    Uses the LLM to determine if a query is meaningful for call records analytics. Returns True if meaningful, False if not, or None if LLM fails.
    """
    prompt = (
        "You are an expert assistant for call records analytics. "
        "Given a user query, determine if it is meaningful and relevant for analytics on call records (e.g., about calls, durations, participants, devices, platforms, directions, audio, etc). "
        "If the query is nonsensical, irrelevant, or not answerable from call records data, respond with 'False'. "
        "If the query is meaningful and relevant, respond with 'True'. "
        "Respond with only 'True' or 'False'.\n"
        f"User query: {query}\n"
    )
    try:
        response = llm_pipeline(prompt, max_new_tokens=10, temperature=0.0)
        answer = response.strip().lower()
        if 'true' in answer:
            return True
        if 'false' in answer:
            return False
        # Fallback: treat as not meaningful if unclear
        return None
    except Exception as e:
        print(f"LLM error in is_meanful_query_llm: {e}")
        return None

def safe_llm_call(prompt, llm_pipeline, timeout=10, **kwargs):
    """Call the LLM with a timeout. Returns None if it fails or times out."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(llm_pipeline, prompt, **kwargs)
        try:
            return future.result(timeout=timeout)
        except Exception as e:
            print(f"LLM call failed or timed out: {e}")
            return None

def process_query(user_query, df, documents, metadatas, llm_pipeline, schema_fields, semantic_search_fn, cache=None):
    """
    Unified query processor for the RAG pipeline.
    Returns a dict with llm_summary, top_snippets, confidence, and explanation.
    """
    # 1. Check for meaningfulness
    if is_meaningless_query(user_query, llm_pipeline):
        return {
            "llm_summary": "Meaningless query. Please ask about call records, participants, devices, platforms, etc.",
            "top_snippets": [],
            "confidence": 0.0,
            "explanation": "Query is not meaningful for call records analytics."
        }

    # 2. Classify intent
    intent = classify_query_intent(user_query, llm_pipeline)
    query_type = intent.get("query_type", "uncertain")

    # 3. Route to appropriate pipeline(s)
    agg_result = None
    sem_results = []
    agg_query = None
    llm_summary = None
    top_snippets = []
    confidence = 0.0
    explanation = ""

    # --- Caching logic ---
    cache_key = f"{query_type}:{user_query}"
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    if query_type == "aggregation":
        agg_query = generate_aggregation_query(user_query, llm_pipeline, schema_fields)
        if "error" not in agg_query:
            agg_result = execute_aggregation_query(df, agg_query)
            llm_summary = synthesize_aggregation_result(user_query, agg_result, llm_pipeline, agg_query)
            explanation = "Aggregation query executed."
        else:
            llm_summary = "No relevant data found for your aggregation query."
            explanation = "No aggregation data found."
    elif query_type == "semantic":
        sem_results = semantic_search_fn(documents, user_query)
        if sem_results:
            top_snippets = [doc for doc, score, meta in sem_results[:3]]
            confidence = float(max([score for doc, score, meta in sem_results[:3]])) if sem_results else 0.0
            # LLM summary over top snippets
            prompt = f"Summarize the following call records for the query: '{user_query}'.\n" + "\n".join(top_snippets)
            llm_summary = safe_llm_call(prompt, llm_pipeline, timeout=15) or "[LLM unavailable, showing top snippets only]"
            explanation = "LLM summary generated from top semantic search results."
        else:
            llm_summary = "No relevant semantic information found."
            explanation = "No semantic data found."
    else:  # 'uncertain' or ambiguous: run both
        agg_query = generate_aggregation_query(user_query, llm_pipeline, schema_fields)
        if "error" not in agg_query:
            agg_result = execute_aggregation_query(df, agg_query)
            llm_summary = synthesize_aggregation_result(user_query, agg_result, llm_pipeline, agg_query)
            explanation = "Both aggregation and semantic search executed."
        sem_results = semantic_search_fn(documents, user_query)
        if sem_results:
            top_snippets = [doc for doc, score, meta in sem_results[:3]]
            confidence = float(max([score for doc, score, meta in sem_results[:3]])) if sem_results else 0.0
            # Optionally, append LLM summary over top snippets
            prompt = f"Summarize the following call records for the query: '{user_query}'.\n" + "\n".join(top_snippets)
            sem_llm_summary = safe_llm_call(prompt, llm_pipeline, timeout=15)
            if sem_llm_summary:
                llm_summary = f"{llm_summary}\n[Semantic summary:] {sem_llm_summary}"
        if not llm_summary:
            llm_summary = "No relevant data found for your query."
            explanation = "No data found."

    result = {
        "llm_summary": llm_summary,
        "top_snippets": top_snippets,
        "confidence": confidence,
        "explanation": explanation
    }
    if cache is not None:
        cache[cache_key] = result
    return result
