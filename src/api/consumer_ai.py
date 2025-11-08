#!/usr/bin/env python
import os
import sys
import json
import pika
from pika.adapters.blocking_connection import BlockingChannel

# Import AI model prediction utilities
from src.models.model_2.predict_input import predict_from_json
from src.api.post import MisinformationReport, MisinfoState, post_from_json


def main():
    print("Initializing AI model...")

    def on_new_post(channel: BlockingChannel, method, properties, body: str):
        try:
            # Deserialize the incoming message
            post = post_from_json(body)
            print(f"[INFO] Received post ID {post.id} from user '{post.username}'")

            # Prepare model input JSON
            json_input = json.dumps({
                "id": str(post.id),
                "text": post.message,
                "date": post.submitted_date.isoformat()
            })

            # Run prediction
            result = predict_from_json(json_input)
            label = int(result["predicted_label"])  # 0 = true, 1 = fake
            confidence = float(result.get("confidence", 0.0))

            # Determine final state
            state = MisinfoState.FAKE if label == 1 else MisinfoState.TRUE

            # Build report including confidence and submission date
            report = MisinformationReport(
                post_id=post.id,
                misinfo_state=state,
                confidence=confidence,
                date_submitted=post.submitted_date
            )

            report_json = json.dumps(report.to_dict(), ensure_ascii=False)
            print(f"[INFO] Completed analysis for post {post.id} → {state.name} (conf={confidence:.3f})")

            # Acknowledge message and publish output
            channel.basic_ack(delivery_tag=method.delivery_tag)
            channel.basic_publish(
                exchange="",
                routing_key="misinfo/output",
                body=report_json
            )
            print("[INFO] Report sent to 'misinfo/output'")

        except Exception as e:
            print(f"[ERROR] Failed to process message: {e}")
            print(f"[DEBUG] Raw message body: {body}")
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    # Connect to RabbitMQ
    print("Connecting to RabbitMQ...")
    with pika.BlockingConnection(pika.ConnectionParameters("localhost")) as connection:
        with connection.channel() as channel:
            channel.queue_declare(queue="misinfo/input")
            channel.queue_declare(queue="misinfo/output")

            channel.basic_consume(
                queue="misinfo/input",
                on_message_callback=on_new_post
            )

            print("Listening on 'misinfo/input' - waiting for messages...")
            channel.start_consuming()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
        sys.exit(0)

