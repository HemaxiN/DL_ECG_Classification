import datetime
import socket
import traceback

import zulip
from loguru import logger

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class FhpLogger:
    def __init__(self, user_id, config_file_path, msg_type="private", to=(), topic="stream events") -> None:
        self.client = zulip.Client(config_file=config_file_path)  # "~/zuliprc-old"
        self.user_id = user_id  # "tomas.pereira@aicos.fraunhofer.pt"
        if msg_type not in ["stream", "private"]:
            raise Exception("type_msg should be stream or private")

        self.msg_type = msg_type
        self.log_receiver = list(to) if len(list(to)) > 0 else [self.user_id]

        if msg_type == "stream":
            self.msg_request = {
                "type": self.msg_type,
                "to": self.log_receiver,
                "topic": topic,
                "content": "",
            }
        else:
            self.msg_request = {
                "type": self.msg_type,
                "to": self.log_receiver,
                "content": "",
            }

    def send_message(self, msg):
        self.msg_request["content"] = msg
        self.client.send_message(self.msg_request)

    def train_logger(self, func):
        def inner1(*args, **kwargs):

            start_time = datetime.datetime.now()
            host_name = socket.gethostname()
            func_name = func.__name__

            msg_contents = [
                "Your training has started 🎬",
                "Machine name: %s" % host_name,
                "Main call: %s" % func_name,
                "Starting date: %s" % start_time.strftime(DATE_FORMAT),
            ]

            self.send_message("\n".join(msg_contents))
            logger.info("Lets start training 🎬")

            try:

                # getting the returned value
                returned_value = func(*args, **kwargs)
                end_time = datetime.datetime.now()
                elapsed_time = end_time - start_time
                msg_contents = [
                    "Your training is complete 🎉",
                    "Machine name: %s" % host_name,
                    "Main call: %s" % func_name,
                    "Starting date: %s"
                    % start_time.strftime(
                        DATE_FORMAT,
                    ),
                    "End date: %s"
                    % end_time.strftime(
                        DATE_FORMAT,
                    ),
                    "Training duration: %s" % str(elapsed_time),
                ]

                self.send_message("\n".join(msg_contents))
                logger.info("Finished training 🎬")

            except Exception as ex:
                end_time = datetime.datetime.now()
                elapsed_time = end_time - start_time
                msg_contents = [
                    "Your training has crashed ☠️",
                    "Machine name: %s" % host_name,
                    "Main call: %s" % func_name,
                    "Starting date: %s"
                    % start_time.strftime(
                        DATE_FORMAT,
                    ),
                    "Crash date: %s" % end_time.strftime(DATE_FORMAT),
                    "Crashed training duration: %s\n\n"
                    % str(
                        elapsed_time,
                    ),
                    "Here's the error:",
                    "%s\n\n" % ex,
                    "Traceback:",
                    "%s" % traceback.format_exc(),
                ]

                self.send_message("\n".join(msg_contents))
                raise ex

            # returning the value to the original frame
            return returned_value

        return inner1
