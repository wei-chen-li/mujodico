"""Run PLF-V (handoff_2026-11-22/scripts/lfddp_v.py, unmodified) on the benchmark tasks:
registers bench_tasks in benchmark_ippas._EXTRA_TASKS, then executes lfddp_v.py as __main__.
usage: [FFM=0.05] plfv.py <Task> <a:b> <label> <out.jsonl> [lfddp_v flags]   (FFM: planning far-field margin, default 0.01)"""
import os, sys, runpy
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LF = os.path.join(HERE, '..', '..', 'handoff_2026-11-22', 'scripts', 'lfddp_v.py')
if os.environ.get('LFV') == 'bench':   # the bench copy (open-loop forward-pass option)
  LF = os.path.join(HERE, 'lfddp_v_bench.py')
sys.argv = [os.path.abspath(LF)] + sys.argv[1:]
if '--par-ls' in sys.argv and sys.argv[sys.argv.index('--par-ls') + 1] != '0':   # as lfddp_v.py, before jax import
  os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') + ' --xla_force_host_platform_device_count=16').strip()
if os.environ.get('FULLLAW') == '1':   # lift every constraint force (fulllaw_patch v2); revertible: unset FULLLAW
  os.environ['IPPAS_EXPORT'] = ''      # the on-disk export cache is keyed on the package source, not on this patch
  import fulllaw_patch
  fulllaw_patch.apply()
  print('fulllaw patch v2 applied', flush=True)
import bench_tasks  # noqa: registers the tasks
if os.environ.get('FFM'):   # planning-model far-field margin override (process-local; the true-law judge does not use LMS_KW)
  sys.path.insert(0, os.path.dirname(os.path.abspath(LF)))
  import reloc_lms
  reloc_lms.LMS_KW['far_field_margin'] = float(os.environ['FFM'])
  print('far_field_margin override:', reloc_lms.LMS_KW['far_field_margin'], flush=True)
runpy.run_path(os.path.abspath(LF), run_name='__main__')
