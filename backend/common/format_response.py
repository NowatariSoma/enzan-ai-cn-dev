from rest_framework.response import Response
from typing import Optional, Union, List, Dict


def standard_response(
    message: str,
    status_code: int,
    data: Optional[Union[List, Dict]] = None,
    error_code: Optional[str] = None,
    error_messages: Optional[Dict[str, List[str]]] = None,
    meta: Optional[Dict] = None,
):
    response = {
        "message": message,
        "status_code": status_code,
    }

    if data is not None:
        response["data"] = data

    if error_code or error_messages:
        response["error"] = {
            "code": error_code,
            "messages": error_messages,
        }

    if meta is not None:
        response["meta"] = meta

    return Response(response, status=status_code)


def create_response(
    message: str,
    status_code: int,
    data: Optional[Union[List, Dict]] = None,
    error_code: Optional[str] = None,
    error_messages: Optional[Dict[str, List[str]]] = None,
    meta: Optional[Dict] = None,
):
    return standard_response(
        message, status_code, data, error_code, error_messages, meta
    )
