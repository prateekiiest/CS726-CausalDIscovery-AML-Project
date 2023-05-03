import numpy as np
import torch


class Validation(object):
    """
    A class for Parameters Validation

    Check whether the parameters are valid,
    including parameter types and parameter values.
    """

    @staticmethod
    def validate_type(x, valid_type):
        """
        Check whether an object is an instance of `valid_type`.

        Parameters
        ----------
        x: object
            object to be verified
        valid_type: type or tuple of type
            A tuple, as in ``validate_type(x, (A, B, ...))``, may be given as
            the target to check against. This is equivalent to
            ``validate_type(x, A) or  validate_type(x, B) or ...`` etc.

        Returns
        -------
        out: True or raise TypeError
        """

        if isinstance(x, valid_type):
            pass
        else:
            raise TypeError(f'Expected type of {x} is an instance '
                            f'of {valid_type}, but got ``{type(x)}``.')

    @staticmethod
    def validate_value(x, valid_value):
        """
        Check whether an object's value is one of `valid_type`.

        Parameters
        ----------
        x: object
            object to be verified
        valid_value: tuple, list
            A tuple, as in ``validate_value(x, (A, B, ...))``, may be given as
            the target to check against. This is equivalent to
            ``validate_value(x, A) or validate_value(x, B) or ...`` etc.

        Returns
        -------
        out: True or raise TypeError
        """

        if x in valid_value:
            pass
        else:
            raise ValueError(f'Expected `x` is one of {valid_value}, '
                             f'but got ``{x}``.')

    @staticmethod
    def to_device(*args, device=None):
        """transfer all of ``args`` to ``device``"""

        out = []
        for each in args:
            if isinstance(each, np.ndarray):
                each = torch.tensor(each, device=device)
            elif isinstance(each, torch.Tensor):
                each = each.to(device=device)
            else:
                raise TypeError(f"Expected type of the args is ``np.ndarray` "
                                f"or ``torch.Tensor``, "
                                f"but got ``{type(each)}``.")
            out.append(each)
        if len(out) > 1:
            return tuple(out)
        else:
            return out[0]

