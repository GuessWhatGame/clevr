from clevr.models.baseline_network import CLEVRNetwork
from clevr.models.film_network import FiLMCLEVRNetwork


# stupid factory class to create networks

def create_network(config, num_words, num_answers, reuse=False, device=''):

    network_type = config["type"]

    if network_type == "film":
        return FiLMCLEVRNetwork(config, num_words=num_words, num_answers=num_answers, reuse=reuse, device=device)
    elif network_type == "baseline":
        return CLEVRNetwork(config, num_words=num_words, num_answers=num_answers, reuse=reuse, device=device)
    else:
        assert False, "Invalid network_type: should be: film/cbn"


