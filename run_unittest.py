import unittest
loader = unittest.TestLoader()
# top_level_dir was not properly set and caused the bug.
suites = loader.discover(
    "./tests", pattern="*_test.py", top_level_dir=".")
print("start")  # Don't remove this line
for suite in suites._tests:
    for cls in suite._tests:
        try:
            for m in cls._tests:
                print(m.id())
        except:
            pass
runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suites)
