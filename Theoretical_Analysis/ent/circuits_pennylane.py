import pennylane as qml

N_QUBITS = 6


NN_PAIRS = [(0, 1), (2, 3), (4, 5), (1, 2), (3, 4)]
CB_PAIRS = [(5, 0), (4, 5), (3, 4), (2, 3), (1, 2), (0, 1)]
def _build_AA_pairs():
    pairs = []
    for ctrl in reversed(range(N_QUBITS)):
        for tgt in range(N_QUBITS):
            if tgt == ctrl:
                continue
            pairs.append((ctrl, tgt))
    return pairs
AA_PAIRS = _build_AA_pairs()


def _S_layer_ref(weights, offset):
    for q in range(N_QUBITS):
        qml.RX(weights[offset + q], wires=q)
    offset += N_QUBITS
    for q in range(N_QUBITS):
        qml.RZ(weights[offset + q], wires=q)
    offset += N_QUBITS
    return offset

def _U_layer_cand(weights, offset):
    for q in range(N_QUBITS):
        qml.RZ(weights[offset + q], wires=q)
    offset += N_QUBITS
    for q in range(N_QUBITS):
        qml.RY(weights[offset + q], wires=q)
    offset += N_QUBITS
    for q in range(N_QUBITS):
        qml.RZ(weights[offset + q], wires=q)
    offset += N_QUBITS
    return offset


def _ref_NN_block(weights, offset):
    for (c, t) in NN_PAIRS:
        qml.CRX(weights[offset], wires=[c, t]); offset += 1
    return offset

def _ref_CB_block(weights, offset):
    for (c, t) in CB_PAIRS:
        qml.CRX(weights[offset], wires=[c, t]); offset += 1
    return offset

def _ref_AA_block(weights, offset):
    for (c, t) in AA_PAIRS:
        qml.CRX(weights[offset], wires=[c, t]); offset += 1
    return offset


def _cand_NN_block():
    for (c, t) in NN_PAIRS:
        qml.CNOT(wires=[c, t])

def _cand_CB_block():
    for (c, t) in CB_PAIRS:
        qml.CNOT(wires=[c, t])

def _cand_AA_block():
    for (c, t) in AA_PAIRS:
        qml.CNOT(wires=[c, t])


def make_ref_NN():
    n_params = 2 * N_QUBITS + 5 * 6 + 2 * N_QUBITS
    def apply(weights):
        offset = _S_layer_ref(weights, 0)
        for _ in range(6):
            offset = _ref_NN_block(weights, offset)
        offset = _S_layer_ref(weights, offset)
        assert offset == n_params, (offset, n_params)
    return n_params, apply

def make_ref_CB():
    n_params = 2 * N_QUBITS + 6 * 6 + 2 * N_QUBITS
    def apply(weights):
        offset = _S_layer_ref(weights, 0)
        for _ in range(6):
            offset = _ref_CB_block(weights, offset)
        offset = _S_layer_ref(weights, offset)
        assert offset == n_params, (offset, n_params)
    return n_params, apply

def make_ref_AA():
    n_params = 2 * N_QUBITS + 30 + 2 * N_QUBITS
    def apply(weights):
        offset = _S_layer_ref(weights, 0)
        offset = _ref_AA_block(weights, offset)
        offset = _S_layer_ref(weights, offset)
        assert offset == n_params, (offset, n_params)
    return n_params, apply

def make_cand_NN():
    n_params = 3 * N_QUBITS + 3 * N_QUBITS
    def apply(weights):
        offset = _U_layer_cand(weights, 0)
        for _ in range(6):
            _cand_NN_block()
        offset = _U_layer_cand(weights, offset)
        assert offset == n_params, (offset, n_params)
    return n_params, apply

def make_cand_CB():
    n_params = 3 * N_QUBITS + 3 * N_QUBITS
    def apply(weights):
        offset = _U_layer_cand(weights, 0)
        for _ in range(5):
            _cand_CB_block()
        offset = _U_layer_cand(weights, offset)
        assert offset == n_params, (offset, n_params)
    return n_params, apply

def make_cand_AA():
    n_params = 3 * N_QUBITS + 3 * N_QUBITS
    def apply(weights):
        offset = _U_layer_cand(weights, 0)
        _cand_AA_block()
        offset = _U_layer_cand(weights, offset)
        assert offset == n_params, (offset, n_params)
    return n_params, apply


ANSATZ_REGISTRY = {
    "Ref-NN":  make_ref_NN,
    "Ref-CB":  make_ref_CB,
    "Ref-AA":  make_ref_AA,
    "Cand-NN": make_cand_NN,
    "Cand-CB": make_cand_CB,
    "Cand-AA": make_cand_AA,
}

EXPECTED_PARAM_COUNTS = {
    "Ref-NN": 54, "Ref-CB": 60, "Ref-AA": 54,
    "Cand-NN": 36, "Cand-CB": 36, "Cand-AA": 36,
}
