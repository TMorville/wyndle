#!/usr/bin/env python3
"""
org.py - Infer organizational structure from Slack data

This script loads all DMs and channel data from the optimized database
and asks the LLM to return a predefined template for the org structure
(key stakeholders, management, direct reports, etc).
"""

import json
import logging
from typing import Any

from openai import OpenAI

from data.duckdb_database import get_optimized_db

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OrgStructureGenerator:
    """
    Analyzes Slack conversations to infer and generate an organizational structure.
    """

    def __init__(self, model_name: str = "gpt-4"):
        self.client: OpenAI | None = None  # Lazy loaded
        self.model_name = model_name
        self.output_path = "org_structure.json"
        self.db = get_optimized_db(lazy_resolver=True)

    def load_conversations(self) -> list[dict[str, Any]]:
        """Loads all conversation data from optimized database."""
        all_conversations: list[dict[str, Any]] = []

        # Get all conversations from database
        conversations = self.db.list_conversations()

        for conv in conversations:
            # Get messages for this conversation (limit to recent for analysis)
            messages = self.db.get_messages(
                conversation_name=conv["name"],
                limit=500,  # Limit to recent messages for analysis
            )

            if messages:
                all_conversations.append(
                    {
                        "meta": {
                            "name": conv["name"],
                            "type": conv["type"],
                            "participants": conv.get("participants", []),
                        },
                        "messages": messages,
                    }
                )

        logger.info(f"Loaded {len(all_conversations)} conversations with messages")
        return all_conversations

    def format_conversations_for_llm(self, conversations: list[dict[str, Any]]) -> str:
        """Formats conversation data into a single string for the LLM prompt."""
        formatted_text: list[str] = []
        for conv_data in conversations:
            conv = conv_data["meta"]
            messages = conv_data["messages"]

            if conv["type"] == "channel":
                formatted_text.append(f"\n=== CHANNEL: {conv['name']} ===")
            else:
                participants = conv.get("participants", [])
                formatted_text.append(f"\n=== DM: {' & '.join(participants)} ===")

            for msg in messages:
                if isinstance(msg, dict) and "user_name" in msg and "clean_text" in msg:
                    # Use clean_text instead of text for better formatting
                    text = msg["clean_text"] or msg.get("text", "")
                    if text:
                        formatted_text.append(f"{msg['user_name']}: {text[:200]}")

        return "\n".join(formatted_text)

    def get_system_prompt(self) -> str:
        """Returns the system prompt with the desired JSON template."""
        return """
You are an expert organizational analyst. Based on the Slack conversation data
provided, analyze the organizational structure and return a JSON response
following this exact template:

{
  "organization": {
    "company_name": "Company Name (if identifiable)",
    "total_people_identified": "Number of unique individuals mentioned"
  },
  "leadership": {
    "executives": [
      {
        "name": "Name",
        "role": "CEO/CTO/etc.",
        "reports_to": null,
        "confidence": "high/medium/low"
      }
    ],
    "management": [
      {
        "name": "Name",
        "role": "Engineering Manager/Product Manager/etc",
        "team": "Team name if identifiable",
        "reports_to": "Executive or higher-level manager",
        "confidence": "high/medium/low"
      }
    ]
  },
  "teams": [
    {
      "name": "Team Name (e.g., Data Squad, Platform)",
      "type": "Engineering/Product/Etc.",
      "members": [
        {"name": "Member Name", "role": "Software Engineer"}
      ],
      "lead": "Team Lead or Manager Name",
      "confidence": "high/medium/low"
    }
  ],
  "insights": {
    "company_culture": "Brief description of company culture based on "
    "communication patterns",
    "communication_patterns": "Description of how teams communicate",
    "key_projects": ["Project names mentioned frequently"],
    "departments_identified": ["Engineering", "Product", "Data", "etc."]
  }
}

Important guidelines:
- Only include people and relationships you can confidently identify from the
  conversations
- Use "high" confidence for clear evidence, "medium" for reasonable inference,
  "low" for speculation
- Focus on actual organizational relationships, not just who talks to whom
- Look for patterns in language that indicate hierarchy (e.g., "my team",
  "reports to", "manager", "lead")
- Identify teams based on shared projects, similar responsibilities, or explicit
  mentions
- Pay attention to channel names and participants to infer team structures
"""

    def generate(self) -> None:
        """
        Main method to load data, prompt the LLM, and save the org structure.
        """
        logger.info("Loading conversations from optimized database...")
        conversations = self.load_conversations()
        if not conversations:
            logger.error("No conversation data found to analyze.")
            return

        logger.info("Formatting data for LLM...")
        conversation_data = self.format_conversations_for_llm(conversations)

        system_prompt = self.get_system_prompt()
        user_prompt = f"""Analyze the following Slack conversation data:

{conversation_data[:50000]}

Please provide a comprehensive organizational analysis following the JSON
template provided in the system prompt."""

        logger.info(f"Sending request to {self.model_name}...")
        try:
            if self.client is None:
                self.client = OpenAI()
            response = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            if response.choices and response.choices[0].message.content:
                org_structure_text = response.choices[0].message.content
                org_structure = json.loads(org_structure_text)

                with open(self.output_path, "w") as f:
                    json.dump(org_structure, f, indent=2)
                logger.info(
                    "Successfully generated and saved org structure to %s",
                    self.output_path,
                )
                self._print_summary(org_structure)
            else:
                logger.error("LLM response did not contain a text block.")

        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from LLM response.")
            if "org_structure_text" in locals():
                logger.debug("LLM Response Text:\n%s", org_structure_text)
        except Exception as e:
            logger.error(f"An error occurred during LLM request: {e}")

    def _print_summary(self, org_structure: dict[str, Any]) -> None:
        """Print a summary of the organizational analysis"""
        summary = [
            "\n" + "=" * 60,
            "ORGANIZATIONAL STRUCTURE ANALYSIS SUMMARY",
            "=" * 60,
        ]

        org_info = org_structure.get("organization", {})
        summary.append(f"Company: {org_info.get('company_name', 'Unknown')}")
        summary.append(
            f"Total People Identified: {org_info.get('total_people_identified', 0)}"
        )

        leadership = org_structure.get("leadership", {})
        executives = leadership.get("executives", [])
        management = leadership.get("management", [])

        if executives:
            summary.append(f"\nExecutives ({len(executives)}):")
            for executive in executives:
                summary.append(
                    f"  - {executive.get('name', 'Unknown')}: "
                    f"{executive.get('role', 'Unknown')} "
                    f"(confidence: {executive.get('confidence', 'unknown')})"
                )

        if management:
            summary.append(f"\nManagers ({len(management)}):")
            for mgr in management:
                summary.append(
                    f"  - {mgr.get('name', 'Unknown')}: {mgr.get('role', 'Unknown')} "
                    f"(confidence: {mgr.get('confidence', 'unknown')})"
                )

        teams = org_structure.get("teams", [])
        if teams:
            summary.append(f"\nTeams ({len(teams)}):")
            for team in teams:
                summary.append(
                    f"  - {team.get('name', 'Unknown')} "
                    f"({team.get('type', 'Unknown')}): "
                    f"{len(team.get('members', []))} members"
                )

        insights = org_structure.get("insights", {})
        departments = insights.get("departments_identified", [])
        if departments:
            summary.append(f"\nDepartments: {', '.join(departments)}")

        projects = insights.get("key_projects", [])
        if projects:
            summary.append(f"Key Projects: {', '.join(projects[:5])}")

        summary.append("\nFull analysis saved to org_structure.json")
        logger.info("\n".join(summary))


def main() -> None:
    """Entry point for the script."""
    generator = OrgStructureGenerator()
    generator.generate()


if __name__ == "__main__":
    main()
