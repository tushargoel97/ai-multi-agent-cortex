import { redirect } from "next/navigation";

export default function LocalModelsRedirect() {
  redirect("/admin?tab=local");
}
