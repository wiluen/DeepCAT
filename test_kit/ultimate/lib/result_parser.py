import re
from pathlib import Path

available_tester = {
    'hibench': re.compile(r'Duration\(s\), (\d+(?:\.\d+)?)')
}

def parse_result(tester_name, result_dir, task_id, rep, printer):
  assert tester_name in available_tester, f'{tester_name} not available'

  result_file = result_dir / f'{task_id}_run_result_{rep}'
  regexp = available_tester['hibench']

  if result_file.is_file():
    content = result_file.read_text()
    match = regexp.findall(content)    # research
    for i in range(len(match)):
      match[i]=float(match[i])

    if match is not None:
      if tester_name == 'hibench':
        mean = sum(match)/len(match)
        return round(mean,2)
      else:
        printer(f'{task_id} - {rep}: Error. Result Error')

    else:
      printer(f'{task_id} - {rep}: WARNING. Result not match.')
  else:
    printer(f'{task_id} - {rep}: WARNING. Result file not found.')
  return None
