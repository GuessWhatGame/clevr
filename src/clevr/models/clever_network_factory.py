from clevr.models.clevr_cbn_network import CBN_CLEVRNetwork
from clevr.models.clevr_film_network import FiLM_CLEVRNetwork


# stupid factory class to create networks

def create_network(config, no_words, no_answers, reuse=False, device=''):

    network_type = config["type"]

    if network_type == "film":
        return FiLM_CLEVRNetwork(config, no_words, no_answers, reuse, device)
    elif network_type == "cbn":
        return CBN_CLEVRNetwork(config, no_words, no_answers, reuse, device)
    else:
        assert False, "Invalid network_type: should be: film/cbn"


