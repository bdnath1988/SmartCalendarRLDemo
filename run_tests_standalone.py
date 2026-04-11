import traceback
from tests.test_grader import (
    test_grader_easy_success,
    test_grader_hard_fail_overlaps,
    test_grader_medium_proportional,
    test_grader_hard_spacing_violations,
    test_grader_hard_perfect
)

tests = [
    test_grader_easy_success,
    test_grader_hard_fail_overlaps,
    test_grader_medium_proportional,
    test_grader_hard_spacing_violations,
    test_grader_hard_perfect
]

passed = 0
for test in tests:
    print(f"Running {test.__name__}...")
    try:
        test()
        print(f"  [PASS] {test.__doc__}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {test.__doc__}")
        traceback.print_exc()

print(f"\n--- SCORE: {passed}/{len(tests)} TESTS PASSED ---")
