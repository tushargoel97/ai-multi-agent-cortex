from enum import StrEnum


class Agents(StrEnum):
    ROUTER = "router"
    GENERALIST = "generalist"
    RESEARCHER = "researcher"
    REASONER = "reasoner"
    PROMPT_CACHER = "prompt_cacher"
