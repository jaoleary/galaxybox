"""
Some general use plotting functions
"""
import matplotlib.pyplot as plt

__author__ = ('Joseph O\'Leary', )


def linestyle(sequence=None, dash=None, dot=None, space=None, buffer=False, offset=0):
    """Generate more linetypes from arbitrary `-`,`.` combinations with added
       IDL like linetype shortcuts.

    Parameters
    ----------
    sequence : string
        An arbitary sequence of `-` and `.` to construct linestyles. Default
        behavior acts just like normal matplotlib `linestyle` arguments. Added
        IDl like shortcuts for long dashes `__` and dash-dot-dot-dot `-:`.
        Underscore is a double dash, and semicolon is a tripple dot.(the default is None).
    dash : float
        length of line dashes (the default is None).
    dot : float
        length of line dots (the default is None).
    space : float
        length of blank spaces in lines (the default is None).
    buffer : float
        lenght of trailing buffer space for a given sequence (the default is False).
    offset : float
        Description of parameter `offset` (the default is 0).

    Returns
    -------
    linestyle : tuple
        tuple containing the matplotlib linestyle `offset`, and `onoffseq`

    """
    if (sequence is None) or (sequence == '-') or (sequence == '_'):
        return '-'
    elif sequence == ':':
        return sequence
    else:
        if dash is None:
            dash = plt.rcParamsDefault['lines.dashed_pattern'][0]
            if (sequence.count('.') > 0) or (sequence.count(':') > 0):
                dash = plt.rcParamsDefault['lines.dashdot_pattern'][0]

        if dot is None:
            dot = plt.rcParamsDefault['lines.dotted_pattern'][0]

        if space is None:
            space = plt.rcParamsDefault['lines.dashed_pattern'][1]

        reftype = {}
        reftype['-'] = [dash, space]
        reftype['_'] = [2 * dash, space]
        reftype['.'] = [dot, space]
        reftype[':'] = [dot, space, dot, space, dot, space]
        onoffseq = []
        for i, s in enumerate(sequence):
            onoffseq.extend(reftype[s])
        if buffer:
            onoffseq[-1] = buffer
        return (offset, onoffseq)


def ls(sequence, **kwargs):
    """Shortcut to the linestyle function."""
    return linestyle(sequence, **kwargs)
