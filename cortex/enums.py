from enum import StrEnum


class Agents(StrEnum):
    ROUTER = "router"
    GENERALIST = "generalist"
    RESEARCHER = "researcher"
    REASONER = "reasoner"
    CODER = "coder"
    DEBUGGER = "debugger"
    PROMPT_CACHER = "prompt_cacher"
    SPECIALIST = "specialist"
    SYNTHESIZER = "synthesizer"
    SHOPPING = "shopping"
    BOOKING = "booking"
