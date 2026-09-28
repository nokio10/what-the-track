import unittest

from tests.generator_service_logic_loader import load_generator_service_logic


YandexRequestThrottle = load_generator_service_logic()["YandexRequestThrottle"]


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, delay):
        self.sleeps.append(delay)
        self.now += delay


class YandexRequestThrottleTests(unittest.TestCase):
    def test_first_request_does_not_sleep(self):
        clock = FakeClock()
        throttle = YandexRequestThrottle(
            min_interval_seconds=0.5,
            time_fn=clock.time,
            sleep_fn=clock.sleep,
        )

        delay = throttle.wait()

        self.assertEqual(delay, 0.0)
        self.assertEqual(clock.sleeps, [])

    def test_second_immediate_request_waits_for_min_interval(self):
        clock = FakeClock()
        throttle = YandexRequestThrottle(
            min_interval_seconds=0.5,
            time_fn=clock.time,
            sleep_fn=clock.sleep,
        )

        throttle.wait()
        delay = throttle.wait()

        self.assertEqual(delay, 0.5)
        self.assertEqual(clock.sleeps, [0.5])

    def test_request_after_elapsed_interval_does_not_sleep(self):
        clock = FakeClock()
        throttle = YandexRequestThrottle(
            min_interval_seconds=0.5,
            time_fn=clock.time,
            sleep_fn=clock.sleep,
        )

        throttle.wait()
        clock.now = 0.8
        delay = throttle.wait()

        self.assertEqual(delay, 0.0)
        self.assertEqual(clock.sleeps, [])

    def test_penalize_pushes_next_request_further_into_future(self):
        clock = FakeClock()
        throttle = YandexRequestThrottle(
            min_interval_seconds=0.5,
            time_fn=clock.time,
            sleep_fn=clock.sleep,
        )

        throttle.wait()
        throttle.penalize(1.0)
        delay = throttle.wait()

        self.assertEqual(delay, 1.5)
        self.assertEqual(clock.sleeps, [1.5])


if __name__ == "__main__":
    unittest.main()
