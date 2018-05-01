import logging;
log = logging.getLogger("LearnedEvolution")

def child(name):
    return log.getChild(name);

def basicConfig(**kwargs):
    logging.basicConfig(**kwargs);
