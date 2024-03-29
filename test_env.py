import time
from firasim_client.env import MarkovDecisionProcess

if __name__ == "__main__":
    loop_time = 0.008
    env = MarkovDecisionProcess()
    state = env.reset_random_init_pos()
    
    last = time.time()
    last_packet = None

    while True:
        if (time.time() - last) < loop_time:
            packet = env.vision.read()
            if packet is not None:
                last_packet = packet
        else:
            break
    
    while True:
        time.sleep(0.1)
        next_state, reward, done = env.step([[0.0, 0.0]])
        # print(next_state)
        # print(reward)
        # print(done)