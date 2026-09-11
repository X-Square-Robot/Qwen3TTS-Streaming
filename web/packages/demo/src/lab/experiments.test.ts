import {describe, expect, it} from "vitest";
import {
  createExperimentStartBarrier,
  ExperimentConnectionError,
  ExperimentStartAbortedError,
} from "./experiments";

describe("experiment connection barrier", () => {
  it("keeps response/text start blocked until every lane is ready", async () => {
    const barrier = createExperimentStartBarrier(3);
    let released = false;
    const started = barrier.wait().then(() => { released = true; });

    barrier.ready(0);
    barrier.ready(2);
    await Promise.resolve();
    expect(released).toBe(false);

    barrier.ready(1);
    await started;
    expect(released).toBe(true);
  });

  it("releases exactly once after all lanes report ready", async () => {
    const barrier = createExperimentStartBarrier(2);
    let starts = 0;
    const wait = barrier.wait().then(() => { starts += 1; });
    barrier.ready(0);
    barrier.ready(0);
    await Promise.resolve();
    expect(starts).toBe(0);
    barrier.ready(1);
    await wait;
    expect(starts).toBe(1);
  });

  it("distinguishes the failed connection from lanes aborted by it", async () => {
    const barrier = createExperimentStartBarrier(2);
    const failure = new Error("socket refused");
    const waiting = barrier.wait();
    barrier.fail(1, failure);

    await expect(waiting).rejects.toMatchObject({
      name: "ExperimentStartAbortedError",
      failedLaneId: 1,
      cause: expect.objectContaining({name: "ExperimentConnectionError", laneId: 1}),
    });

    const connectionError = new ExperimentConnectionError(1, failure);
    expect(connectionError).toMatchObject({name: "ExperimentConnectionError", laneId: 1, cause: failure});
    expect(new ExperimentStartAbortedError(connectionError).message).toContain("第 2 路失败");
  });

  it("does not reopen after a connection failure", async () => {
    const barrier = createExperimentStartBarrier(2);
    barrier.fail(0, new Error("offline"));
    barrier.ready(1);
    await expect(barrier.wait()).rejects.toBeInstanceOf(ExperimentStartAbortedError);
  });

  it("unblocks every waiter when a lane is cancelled before connecting", async () => {
    const barrier = createExperimentStartBarrier(2);
    const waiting = barrier.wait();
    barrier.cancel(0);
    await expect(waiting).rejects.toMatchObject({name: "ExperimentStartAbortedError", failedLaneId: 0});
  });
});
