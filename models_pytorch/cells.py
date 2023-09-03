import torch
import torch.nn as nn
import torch.nn.init as init


class Double(nn.Module):
    """
    Double layer architecture.
    """
    def __init__(self, cell_class, input_size, hidden_size, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_1 = cell_class(input_size, int(hidden_size / 2), **kwargs)
        self.cell_2 = cell_class(input_size, int(hidden_size / 2), **kwargs)

    def forward(self, x, h0):
        if h0 is None:
            h01 = None
            h02 = None
        else:
            h01 = tuple(h[:, :int(self.hidden_size / 2)].contiguous() for h in h0)
            h02 = tuple(h[:, int(self.hidden_size / 2):].contiguous() for h in h0)

        x1, hn1 = self.cell_1.forward(x, h01)
        x2, hn2 = self.cell_2.forward(x, h02)

        x = torch.cat((x1, x2), dim=-1)
        hn = tuple(torch.cat((h1.squeeze(0), h2.squeeze(0)), dim=-1) for h1, h2 in zip(hn1, hn2))

        return x, hn


class LSTM(nn.LSTM):
    """
    Long short-term memory.
    """

    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__(input_size=input_size, hidden_size=hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x, h0):
        if h0 is None:
            h0 = torch.zeros(x.size(1), self.hidden_size, device=x.device)
            c0 = torch.zeros(x.size(1), self.hidden_size, device=x.device)
        else:
            h0, c0 = h0

        x, (hn, cn) = super().forward(x, (h0.unsqueeze(0), c0.unsqueeze(0)))
        return x, (hn.squeeze(0), cn.squeeze(0))


class CHRONO(LSTM):
    """
    Chrono-initialised LSTM
    """

    def __init__(self, input_size, hidden_size, **kwargs):

        if 't_max' not in kwargs:
            raise ValueError('Missing argument t_max for CHRONO cell')

        T_max = kwargs['t_max']

        super().__init__(input_size=input_size, hidden_size=hidden_size)

        parameters = {k: v for k, v in self.named_parameters()}

        for i in range(self.num_layers):
            for g in ('i', 'h'):
                # bias is (b_gi|b_gf|b_gc|b_go) in R^{4*H}
                bias = parameters[f'bias_{g}h_l{i}']
                uniform = torch.rand(self.hidden_size) * (T_max - 2) + 1
                with torch.no_grad():
                    bias[self.hidden_size:2 * self.hidden_size] = uniform.log()
                    bias[:self.hidden_size] = - uniform.log()
                    bias[2 * self.hidden_size:] = torch.zeros(2 * self.hidden_size)
                # init.zeros_(bias[2 * self.hidden_size:])


class GRU(nn.GRU):
    """
    Gated recurrent unit.
    """

    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__(input_size=input_size, hidden_size=hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                for i in range(3):
                    mul = param.shape[0] // 3
                    init.xavier_uniform_(param[i * mul:(i + 1) * mul])
            elif 'weight_hh' in name:
                for i in range(3):
                    mul = param.shape[0] // 3
                    init.orthogonal_(param[i * mul:(i + 1) * mul])
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x, h0):
        if h0 is None:
            h0 = torch.zeros(x.size(1), self.hidden_size, device=x.device)
        else:
            h0, = h0

        x, hn = super().forward(x, h0.unsqueeze(0))
        return x, (hn.squeeze(0),)


class MGU(nn.Module):
    """
    Minimal gated unit (see arXiv:1603.09420).
    """

    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Forget gate
        Wx_f = init.xavier_uniform_(torch.empty(hidden_size, input_size))
        Wh_f = init.xavier_uniform_(torch.empty(hidden_size, hidden_size))
        self.Wx_f = nn.Parameter(Wx_f)
        self.Wh_f = nn.Parameter(Wh_f)
        self.b_f = nn.Parameter(init.zeros_(torch.empty(hidden_size)))

        # Hidden state
        Wx_h = init.xavier_uniform_(torch.empty(hidden_size, input_size))
        Wh_h = init.xavier_uniform_(torch.empty(hidden_size, hidden_size))
        self.Wx_h = nn.Parameter(Wx_h)
        self.Wh_h = nn.Parameter(Wh_h)
        self.b_h = nn.Parameter(init.zeros_(torch.empty(hidden_size)))

    def forward(self, x_seq, h0):
        if h0 is None:
            h0 = torch.zeros(x_seq.size(1), self.hidden_size,
                             device=x_seq.device)
        else:
            h0, = h0

        assert h0.size(0) == x_seq.size(1)
        assert h0.size(1) == self.hidden_size
        assert x_seq.size(2) == self.input_size

        seq_len = x_seq.size(0)
        batch_size = x_seq.size(1)

        y_seq = torch.empty(seq_len, batch_size, self.hidden_size,
                            device=x_seq.device)

        h = h0
        for t in range(seq_len):
            x = x_seq[t, :, :]

            f = torch.sigmoid(torch.mm(x, self.Wx_f.T) +
                              torch.mm(h, self.Wh_f.T) +
                              self.b_f)
            h_t = torch.tanh(torch.mm(f * h, self.Wh_h.T) +
                             torch.mm(x, self.Wx_h.T) + self.b_h)
            h = (1.0 - f) * h + f * h_t

            y_seq[t, ...] = h

        return y_seq, (h,)


class BRC(nn.Module):
    """
    Bistable recurrent cell (see arXiv:2006.05252).
    """

    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Forget gate
        U_c = init.xavier_uniform_(torch.empty(hidden_size, input_size))
        self.U_c = nn.Parameter(U_c)
        self.w_c = nn.Parameter(init.constant_(torch.empty(hidden_size), 1.0))
        self.b_c = nn.Parameter(init.zeros_(torch.empty(hidden_size)))

        # Reset gate
        U_a = init.xavier_uniform_(torch.empty(hidden_size, input_size))
        self.U_a = nn.Parameter(U_a)
        self.w_a = nn.Parameter(init.constant_(torch.empty(hidden_size), 1.0))
        self.b_a = nn.Parameter(init.zeros_(torch.empty(hidden_size)))

        # Hidden state
        U_h = init.xavier_uniform_(torch.empty(hidden_size, input_size))
        self.U_h = nn.Parameter(U_h)
        self.b_h = nn.Parameter(init.zeros_(torch.empty(hidden_size)))

    def forward(self, x_seq, h0):
        if h0 is None:
            h0 = torch.zeros(x_seq.size(1), self.hidden_size,
                             device=x_seq.device)
        else:
            h0, = h0

        assert h0.size(0) == x_seq.size(1)
        assert h0.size(1) == self.hidden_size
        assert x_seq.size(2) == self.input_size

        seq_len = x_seq.size(0)
        batch_size = x_seq.size(1)

        y_seq = torch.empty(seq_len, batch_size, self.hidden_size,
                            device=x_seq.device)

        h = h0
        for t in range(seq_len):
            x = x_seq[t, :, :]
            c = torch.sigmoid(torch.mm(x, self.U_c.T) + self.w_c * h +
                              self.b_c)
            a = 1. + torch.tanh(torch.mm(x, self.U_a.T) + self.w_a * h +
                                self.b_a)
            h = c * h + (1. - c) * torch.tanh(torch.mm(x, self.U_h.T) +
                                              a * h + self.b_h)
            y_seq[t, ...] = h

        return y_seq, (h,)


class nBRC(nn.Module):
    """
    Recurrently neuromodulated bistable recurrent cell (see arXiv:2006.05252).
    """

    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Forget gate
        U_c = init.xavier_uniform_(torch.empty(hidden_size, input_size))
        self.U_c = nn.Parameter(U_c)
        W_c = init.xavier_uniform_(torch.empty(hidden_size, hidden_size))
        self.W_c = nn.Parameter(W_c)
        self.b_c = nn.Parameter(init.zeros_(torch.empty(hidden_size)))

        # Reset gate
        U_a = init.xavier_uniform_(torch.empty(hidden_size, input_size))
        self.U_a = nn.Parameter(U_a)
        W_a = init.xavier_uniform_(torch.empty(hidden_size, hidden_size))
        self.W_a = nn.Parameter(W_a)
        self.b_a = nn.Parameter(init.zeros_(torch.empty(hidden_size)))

        # Hidden state
        U_h = init.xavier_uniform_(torch.empty(hidden_size, input_size))
        self.U_h = nn.Parameter(U_h)
        self.b_h = nn.Parameter(init.zeros_(torch.empty(hidden_size)))

    def forward(self, x_seq, h0):
        if h0 is None:
            h0 = torch.zeros(x_seq.size(1), self.hidden_size,
                             device=x_seq.device)
        else:
            h0, = h0

        assert h0.size(0) == x_seq.size(1)
        assert h0.size(1) == self.hidden_size
        assert x_seq.size(2) == self.input_size

        seq_len = x_seq.size(0)
        batch_size = x_seq.size(1)

        y_seq = torch.empty(seq_len, batch_size, self.hidden_size,
                            device=x_seq.device)

        h = h0
        for t in range(seq_len):
            x = x_seq[t, :, :]
            c = torch.sigmoid(torch.mm(x, self.U_c.T) +
                              torch.mm(h, self.W_c.T) + self.b_c)
            a = 1. + torch.tanh(torch.mm(x, self.U_a.T) +
                                torch.mm(h, self.W_a.T) + self.b_a)
            h = c * h + (1. - c) * torch.tanh(torch.mm(x, self.U_h.T) +
                                              a * h + self.b_h)
            y_seq[t, ...] = h

        return y_seq, (h,)
