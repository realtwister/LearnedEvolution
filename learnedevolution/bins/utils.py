def confirm(question, should_confirm, before_fn = None):
    if not should_confirm:
        if before_fn is not None:
            before_fn();
        return True;
    while True:
        if before_fn is not None:
            before_fn();
        answer = input(question +" (y/n) ");
        if answer.lower() in ["y", "yes"]:
            return True;
        if answer.lower() in ["n", "no"]:
            return False;
