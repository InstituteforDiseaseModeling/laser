import ast
import inspect
import textwrap
from functools import lru_cache

from numba import njit
from numba import prange


# --- AST extraction helpers ---

def extract_predicate_action_bodies(predicate_fn, action_fn):
    pred_src = textwrap.dedent(inspect.getsource(predicate_fn))
    act_src = textwrap.dedent(inspect.getsource(action_fn))

    pred_ast = ast.parse(pred_src)
    act_ast = ast.parse(act_src)

    pred_func = pred_ast.body[0]
    act_func = act_ast.body[0]

    if len(pred_func.body) != 1 or not isinstance(pred_func.body[0], ast.Return):
        raise ValueError("Predicate function must consist of a single 'return' statement.")
    predicate_expr = pred_func.body[0].value

    # We don't want to return out of the loop early, so filter out return statements
    action_body = [stmnt for stmnt in act_func.body if not isinstance(stmnt, ast.Return)]

    return predicate_expr, action_body


def build_fused_apply_fn_ast(predicate_fn, action_fn, arrays_and_consts):
    predicate_expr, action_body = extract_predicate_action_bodies(predicate_fn, action_fn)

    args = [ast.arg(arg="n", annotation=None)] + [ast.arg(arg=name, annotation=None) for name in arrays_and_consts]

    loop_var = ast.Name(id="i", ctx=ast.Store())
    loop_iter = ast.Call(func=ast.Name(id="prange", ctx=ast.Load()), args=[ast.Name(id="n", ctx=ast.Load())], keywords=[])

    if_stmt = ast.If(test=predicate_expr, body=action_body, orelse=[])

    loop = ast.For(target=loop_var, iter=loop_iter, body=[if_stmt], orelse=[])

    func_def = ast.FunctionDef(
        name="apply_fn",
        args=ast.arguments(posonlyargs=[], args=args, vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
        body=[loop],
        # decorator_list=[ast.Name(id="njit", ctx=ast.Load())],
        decorator_list=[
            ast.Call(
                func=ast.Name(id='njit', ctx=ast.Load()),
                args=[],
                keywords=[
                    ast.keyword(arg='parallel', value=ast.Constant(value=True))
                ]
            )
        ],
        returns=None,
    )

    module = ast.Module(body=[func_def], type_ignores=[])
    ast.fix_missing_locations(module)

    return module


def compile_fused_apply_fn(module_ast):
    code = compile(module_ast, filename="<ast>", mode="exec")
    local_env = {"njit": njit, "prange": prange}
    exec(code, local_env)
    return local_env["apply_fn"]


# --- Main builder with caching ---


@lru_cache(maxsize=128)
def build_fused_apply_function(predicate_fn, action_fn):
    pred_params = list(inspect.signature(predicate_fn).parameters.keys())
    act_params = list(inspect.signature(action_fn).parameters.keys())

    # pred_args = [p for p in pred_params if p != "i"]
    # act_args = [p for p in act_params if p != "i"]

    # arrays = []
    # constants = []
    # for p in sorted(set(pred_args + act_args)):
    #     if p in ["position", "velocity", "state"]:  # crude check; improve later
    #         arrays.append(p)
    #     else:
    #         constants.append(p)

    # all_args = arrays + constants
    all_args = [p for p in pred_params if p != "i"] + [p for p in act_params if p not in pred_params]

    fused_ast = build_fused_apply_fn_ast(predicate_fn, action_fn, all_args)
    apply_fn = compile_fused_apply_fn(fused_ast)

    return apply_fn, all_args


# --- Updated apply() function using keyword arguments ---


def apply(predicate_fn, action_fn, count, **kwargs):
    apply_fn, all_arg_names = build_fused_apply_function(predicate_fn, action_fn)

    # # Check that all required arguments are provided
    # missing = [name for name in all_arg_names if name not in kwargs]
    # if missing:
    #     raise ValueError(f"Missing required arguments for apply_fn: {missing}")

    # # Extract arguments in correct order
    # ordered_args = [kwargs[name] for name in all_arg_names]

    # Call the compiled function
    # apply_fn(count, *ordered_args)
    apply_fn(count, *[kwargs[name] for name in all_arg_names])


def numbafy(predicate_fn, action_fn):
    apply_fn, all_arg_names = build_fused_apply_function(predicate_fn, action_fn)

    def numbafied(count, **kwargs):
        # Check that all required arguments are provided
        # missing = [name for name in all_arg_names if name not in kwargs]
        # if missing:
        #     raise ValueError(f"Missing required arguments: {missing}")

        # # Extract arguments in correct order
        # ordered_args = [kwargs[name] for name in all_arg_names]

        # Call the compiled function
        # apply_fn(count, *ordered_args)
        apply_fn(count, *[kwargs[name] for name in all_arg_names])

    return numbafied

