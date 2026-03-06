import { createResource } from '@/lib/actions/resources';
import {
  convertToModelMessages,
  streamText,
  tool,
  UIMessage,
  stepCountIs,
} from 'ai';
import { z } from 'zod';
import { findRelevantContent } from '@/lib/ai/embedding';

// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

export async function POST(req: Request) {
  let messages: UIMessage[];
  try {
    const body = await req.json();
    if (!Array.isArray(body?.messages)) {
      return new Response('Invalid request body: messages must be an array', { status: 400 });
    }
    messages = body.messages;
  } catch {
    return new Response('Invalid request body', { status: 400 });
  }

  let result;
  try {
    result = streamText({
      model: 'openai/gpt-4o',
      messages: convertToModelMessages(messages),
      stopWhen: stepCountIs(10),
      system: `You are a helpful assistant with access to a personal knowledge base.
Always use the getInformation tool to retrieve relevant context before answering.
If the user shares a new piece of knowledge, use the addResource tool to save it immediately without asking for confirmation.
If no relevant information is found in the knowledge base, respond with: "Sorry, I don't know."
Never answer from general knowledge — only use what is retrieved from tools.`,
      tools: {
        addResource: tool({
          description: `add a resource to your knowledge base.
          If the user provides a random piece of knowledge unprompted, use this tool without asking for confirmation.`,
          inputSchema: z.object({
            content: z
              .string()
              .describe('the content or resource to add to the knowledge base'),
          }),
          execute: async ({ content }) => createResource({ content }),
        }),
        getInformation: tool({
          description: `get information from your knowledge base to answer questions.`,
          inputSchema: z.object({
            question: z.string().describe('the users question'),
          }),
          execute: async ({ question }) => findRelevantContent(question),
        }),
      },
    });
  } catch (error) {
    console.error('streamText error:', error);
    return new Response('Internal server error', { status: 500 });
  }

  return result.toUIMessageStreamResponse();
}