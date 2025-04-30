async def serve():
    import logging

    import grpc.aio

    from . import ollama_pb2_grpc
    from .service import OllamaService
    
    logger = logging.getLogger(__name__)
    
    service = OllamaService()
    server = grpc.aio.server(
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024)
        ]
    )
    
    ollama_pb2_grpc.add_OllamaServiceServicer_to_server(service, server)
    server.add_insecure_port("[::]:50051")
    
    await server.start()
    logger.info("Ollama gRPC server started on port 50051")
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        await service.cleanup()
        await server.stop(0)

if __name__ == "__main__":
    asyncio.run(serve())