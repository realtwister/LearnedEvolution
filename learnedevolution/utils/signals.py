import blinker as b;
from .logging import log;

def signal(name):
    return b.signal("structevol."+name);

def method_event(name, before= True, after = True):
    if before:
        before_signal = signal('before_'+name);
    if after:
        after_signal = signal('after_'+name);
    def func_wrapper(fn):
        def wrapped_function(self, *args, **kwargs):
            if before:
                before_signal.send(self, args = args, kwargs = kwargs);
            res = fn(self, *args, **kwargs);
            if after:
                after_signal.send(self, args = args, kwargs = kwargs, result = res);
            return res;
        return wrapped_function
    return func_wrapper

class TimedCallback(object):
    def __init__(self, event, *, sender = b.Signal.ANY, fns=[], once_in = 1, offset = 0, one_shot = False):
        assert isinstance(event, str), "event should be a string";
        assert isinstance(once_in, int) and once_in >= 1, "Once_in should be a natural number";
        assert isinstance(offset, int), "Offset should be an integer";
        assert isinstance(one_shot, bool), "One_shot should be a boolean";
        if callable(fns):
            fns = [fns];
        assert isinstance(fns,list), "fns should be a list of functions";
        for fn in fns:
            assert callable(fn), "Each element of fns should be callable";

        self._signal = signal(event);
        self._fns = fns;
        self._once_in = once_in;
        self._offset = offset;
        self._sender = sender;
        if one_shot:
            self._callback = self._one_shot_callback;
        else:
            self._callback = self._timed_callback;

        self._connected = False;
        self.reset_timer();

    def reset_timer(self, offset = None):
        if offset is not None:
            self._offset = offset;
        self._current_timer = self._offset;
        return self;

    def add_callback(self, fn):
        assert callable(fn), "fn should be callable";
        assert not self._active, "Cannot add callbacks while in event";
        fns.append(fn);
        return self;

    def connect(self):
        if self._connected:
            log.info("Already connected so not doing anything.")
            return self;
        self._signal.connect(self._callback, sender = self._sender, weak = False);
        self._connected = True;
        return self;

    def disconnect(self):
        if not self._connected:
            log.info("Not connected so not doing anything.");
            return self;
        self._signal.disconnect(self._callback, sender = self._sender);
        self._connected = False;
        return self;

    def _one_shot_callback(self, *args, **kwargs):
        if self._connected:
            self._call_callbacks(args,kwargs);
            self.disconnect();
        else:
            log.info("Callback called without being connected.");

    def _timed_callback(self, *args, **kwargs):
        if self._connected:
            if self._current_timer >= 0 and self._current_timer % self._once_in == 0:
                self._call_callbacks(args,kwargs);
            self._current_timer += 1;
        else:
            log.info("Callback called without being connected");

    def _call_callbacks(self, args, kwargs):
        self._active = True;
        for fn in self._fns:
            fn(*args, **kwargs);
        self._active = False;
