import ast
import functools
import inspect


def install_epoch_recovery(module, name, phase, recovery):
    original = getattr(module, name)
    tree = ast.parse(inspect.getsource(original))
    function = tree.body[0]
    loops = [
        node
        for node in function.body
        if isinstance(node, ast.For)
        and ast.dump(node.target) == ast.dump(ast.Name(id="epoch", ctx=ast.Store()))
        and ast.unparse(node.iter) == "range(epochs)"
    ]
    if len(loops) != 1:
        raise RuntimeError(f"Unrecognized runtime {phase} epoch loop; refusing resume patch")
    loop = loops[0]
    index = function.body.index(loop)
    prefix = ast.parse(f"_runtime_losses = __runtime_recovery.restore_epoch({phase!r}, locals())").body
    if phase == "flow":
        starts = [
            j
            for j, node in enumerate(function.body[:index])
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "train_loss_return" for t in node.targets)
        ]
        if len(starts) != 1:
            raise RuntimeError("Unrecognized runtime initial flow loss block")
        start = starts[0]
        branch = ast.parse(
            "if _runtime_losses is None:\n    pass\nelse:\n    train_losses, val_losses = _runtime_losses"
        ).body[0]
        branch.body = function.body[start:index]
        function.body[start:index] = prefix + [branch]
    else:
        function.body[index:index] = (
            prefix
            + ast.parse("if _runtime_losses is not None:\n    train_loss, val_loss = _runtime_losses").body
        )
    loop.iter = ast.parse(f"range(__runtime_recovery.next_epoch({phase!r}), epochs)", mode="eval").body
    loop.body += ast.parse(f"__runtime_recovery.save_epoch({phase!r}, epoch, locals())").body
    ast.fix_missing_locations(tree)
    namespace = {}
    original.__globals__["__runtime_recovery"] = recovery
    exec(
        compile(tree, f"<runtime epoch recovery: {original.__code__.co_filename}>", "exec"),
        original.__globals__,
        namespace,
    )
    patched = functools.update_wrapper(namespace[name], original)
    setattr(module, name, patched)
    return patched
