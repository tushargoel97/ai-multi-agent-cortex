import { redirect } from "next/navigation";

export default function ModelsRedirect() {
  redirect("/admin?tab=models");
}
