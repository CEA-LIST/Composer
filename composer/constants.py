CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15
LOGDIR = "."

IGNORE_INDEX = -100
DEFAULT_TOKENS = {
    'pad': "[PAD]",
    'bos': "<s>",
    'eos': "</s>",
    'unk': "<unk>",
    'sep': "<sep>",
    'boi': "<img>",
    'eoi': "</img>",
    'image': "<image>",
    'ground_and_reason': "<ground_and_reason>"
}

SPECIAL_TOKENS = {

    # Reasoning
    "REASONING_START": "<reasoning>",
    "REASONING_END": "</reasoning>",

    # Answer
    "ANSWER_START": "<answer>",
    "ANSWER_END": "</answer>",
    
    # Misc
    "NO_REGION": "<no_region>", # no region placeholder
    "SEPARATOR": "<separator>", # separator token 

    # Atomic tasks
    "OBJECT_RECOGNITION_START": "<objectRecognition>",
    "OBJECT_RECOGNITION_END": "</objectRecognition>",

    "SPATIAL_POSITION_START": "<spatialPosition>",
    "SPATIAL_POSITION_END": "</spatialPosition>",

    "ATTRIBUTE_COLOR_START": "<attributeColor>",
    "ATTRIBUTE_COLOR_END": "</attributeColor>",

    "SPATIAL_RELATIONSHIP_START": "<spatialRelationship>",
    "SPATIAL_RELATIONSHIP_END": "</spatialRelationship>",
    
    # instances
    "LABEL_START": "<label>", 
    "LABEL_END": "</label>",

    "BBOX_START": "<bbox>",
    "BBOX_END": "</bbox>",

    "COLOR_START": "<color>",
    "COLOR_END": "</color>",

    "RELATION_START": "<relation>",
    "RELATION_END": "</relation>",

    "SPATIAL_POSITION_START": "<spatialPosition>",
    "SPATIAL_POSITION_END": "</spatialPosition>",

    "OBJECT_INSTANCE_START": "<objectInstance>",
    "OBJECT_INSTANCE_END": "</objectInstance>",

    "SUBJECT_INSTANCE_START": "<subjectInstance>",
    "SUBJECT_INSTANCE_END": "</subjectInstance>",

    "ATTRIBUTE_COLOR_INSTANCE_START": "<attributeColorInstance>",
    "ATTRIBUTE_COLOR_INSTANCE_END": "</attributeColorInstance>",

    "RELATION_INSTANCE_START": "<relationInstance>",
    "RELATION_INSTANCE_END": "</relationInstance>",

    "SPATIAL_POSITION_INSTANCE_START": "<spatialPositionInstance>",
    "SPATIAL_POSITION_INSTANCE_END": "</spatialPositionInstance>",   

    # Additional spatial tokens
    "ADDITIONAL_SPATIAL_TOKEN_1": "<additionalSpatialToken1>",
    "ADDITIONAL_SPATIAL_TOKEN_2": "<additionalSpatialToken2>",
    "ADDITIONAL_SPATIAL_TOKEN_3": "<additionalSpatialToken3>",
    "ADDITIONAL_SPATIAL_TOKEN_4": "<additionalSpatialToken4>",
    "ADDITIONAL_SPATIAL_TOKEN_5": "<additionalSpatialToken5>",
    "ADDITIONAL_SPATIAL_TOKEN_6": "<additionalSpatialToken6>",
    "ADDITIONAL_SPATIAL_TOKEN_7": "<additionalSpatialToken7>",
    "ADDITIONAL_SPATIAL_TOKEN_8": "<additionalSpatialToken8>",
    "ADDITIONAL_SPATIAL_TOKEN_9": "<additionalSpatialToken9>",
    "ADDITIONAL_SPATIAL_TOKEN_10": "<additionalSpatialToken10>",
}

N_REGION_TOKENS = 256
PROXY_TOKENS = [f"<r{i}>" for i in range(1, N_REGION_TOKENS + 1)]
