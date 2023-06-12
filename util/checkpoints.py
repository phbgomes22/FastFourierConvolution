import os
import re


def _get_latest_checkpoint(ckpt_dir):
    """
    Given a checkpoint dir, finds the checkpoint with the latest training step.
    """
    def _get_step_number(k):
        """
        Helper function to get step number from checkpoint files.
        """
        search = re.search(r'(\d+)_steps', k)

        if search:
            return int(search.groups()[0])
        else:
            return -float('inf')

    if not os.path.exists(ckpt_dir):
        return None

    files = os.listdir(ckpt_dir)
    if len(files) == 0:
        return None

    ckpt_file = max(files, key=lambda x: _get_step_number(x))

    return os.path.join(ckpt_dir, ckpt_file)
