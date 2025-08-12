from drf_yasg import openapi




def standard_response_schema(
    data_properties: dict = None,
    error_code_example: str = None,
    error_messages_example: dict = None,
    meta_properties: dict = None,
    message_example: str = "Success",
    status_code_example: int = 200,
):
    properties = {
        "message": openapi.Schema(type=openapi.TYPE_STRING, example=message_example),
        "status_code": openapi.Schema(type=openapi.TYPE_INTEGER, example=status_code_example),
    }

    if data_properties is not None:
        properties["data"] = openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties=data_properties,
        )

    if error_code_example or error_messages_example:
        properties["error"] = openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                "code": openapi.Schema(type=openapi.TYPE_STRING, example=error_code_example),
                "messages": openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    additional_properties=openapi.Schema(
                        type=openapi.TYPE_ARRAY,
                        items=openapi.Schema(type=openapi.TYPE_STRING)
                    ),
                    example=error_messages_example,
                ),
            }
        )

    if meta_properties is not None:
        properties["meta"] = openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties=meta_properties,
        )

    return openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties=properties
    )
