# keep running mulitiple experiments

import sys
import asyncio
from os import getcwd
from pathlib import Path

from lib.other import run_playbook


async def run_shell(cmd):
  process = await asyncio.create_subprocess_shell(
      cmd=cmd,
      cwd=getcwd(),
  )
  await process.communicate()


async def main(prefix, limit, cmd):
  rep = 0
  while rep < limit:
    # print(f'{cmd} task_name={prefix}-{rep}')
    # break
    await run_shell(f'pipenv run {cmd} task_name={prefix}-{rep}')
    # if rep > 0:
    #   await run_playbook(
    #       playbook_path=Path(
    #           __file__, '../../../playbooks/cleanup.yml').resolve(),
    #       task_name=f'{prefix}-{rep}',
    #       db_name='mongodb',
    #       host=['m1', 'm2', 'm3', 'm4']
    #   )
    await asyncio.sleep(1)

    rep += 1

_, prefix, limit, *args = sys.argv

limit = int(limit)

loop = asyncio.get_event_loop()
loop.run_until_complete(
    main(prefix, limit, ' '.join(args))
)
loop.close()
