from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,SystemMessage
from typing import Union,List,Optional

def filter_messages(
        messages:List[BaseMessage],
        include_types: Union[str,List[str]]=None,
        exclude_names :List[str]=None,
        exclude_ids:List[str]=None,
):
    if isinstance(include_types,str):
        include_types=[include_types]

    filtered = []
    for msg in messages:
        msg_type = (
        "human" if isinstance(msg,HumanMessage)
        else "ai" if  isinstance(msg,AIMessage)
        else "system" if isinstance(msg,SystemMessage)
        else "other")
        if include_types and msg_type not in include_types:
            continue

        if exclude_names and getattr(msg,"name",None) in exclude_names:
            continue
        
        if exclude_ids and  getattr(msg,"id",None) in exclude_ids:
            continue

        filtered.append(msg)

    return filtered