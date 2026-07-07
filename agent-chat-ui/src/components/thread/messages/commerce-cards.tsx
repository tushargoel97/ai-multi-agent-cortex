import { useState, type ReactNode } from "react";
import { ExternalLink, ShoppingBag, Ticket } from "lucide-react";

/** Host portion of a domain that may carry a path, e.g. "google.com/travel". */
function host(domain: string): string {
  return (domain || "").replace(/^https?:\/\//, "").split("/")[0];
}

/** Human-friendly domain: strip protocol, www, and any path. */
function prettyDomain(domain: string): string {
  return host(domain).replace(/^www\./, "");
}

/** Retailer/platform favicon with a graceful letter-avatar fallback. */
function Favicon({ domain }: { domain: string }) {
  const [failed, setFailed] = useState(false);
  const letter = (prettyDomain(domain)[0] || "?").toUpperCase();

  if (failed) {
    return (
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-muted text-sm font-semibold text-muted-foreground">
        {letter}
      </div>
    );
  }
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={`https://www.google.com/s2/favicons?domain=${encodeURIComponent(host(domain))}&sz=64`}
      alt=""
      width={36}
      height={36}
      className="h-9 w-9 shrink-0 rounded-md bg-muted object-contain p-0.5"
      onError={() => setFailed(true)}
    />
  );
}

function CardHeader({
  icon,
  title,
  region,
}: {
  icon: ReactNode;
  title: string;
  region?: string;
}) {
  return (
    <div className="mb-2 flex items-center gap-2 text-sm font-medium text-foreground">
      {icon}
      <span>{title}</span>
      {region ? (
        <span className="text-xs font-normal text-muted-foreground">
          · {region}
        </span>
      ) : null}
    </div>
  );
}

interface Offer {
  retailer: string;
  title?: string;
  url: string;
  price?: string;
  snippet?: string;
  available?: boolean | null;
}

interface ShoppingData {
  product?: string;
  region?: string;
  currency?: string;
  offers?: Offer[];
  note?: string;
}

export function ShoppingCards({ data }: { data: ShoppingData }) {
  const offers = (data?.offers ?? []).filter((o) => o?.url);

  return (
    <div className="mx-auto w-full max-w-3xl">
      <CardHeader
        icon={<ShoppingBag className="h-4 w-4 text-muted-foreground" />}
        title={data?.product ? `Where to buy ${data.product}` : "Shopping options"}
        region={data?.region}
      />
      {offers.length === 0 ? (
        <p className="rounded-lg border border-dashed border-border px-3 py-4 text-center text-xs text-muted-foreground">
          {data?.note || "No offers found, try a more specific product name."}
        </p>
      ) : (
        <div className="grid gap-2 sm:grid-cols-2">
          {offers.map((o, i) => (
            <a
              key={`${o.url}-${i}`}
              href={o.url}
              target="_blank"
              rel="noopener noreferrer"
              className="group flex items-center gap-3 rounded-lg border border-border bg-background p-3 transition-colors hover:border-foreground/30 hover:bg-muted/40"
            >
              <Favicon domain={o.retailer} />
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-medium text-foreground">
                  {o.title || prettyDomain(o.retailer)}
                </div>
                <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <span className="truncate">{prettyDomain(o.retailer)}</span>
                  {o.available === true ? (
                    <span className="shrink-0 rounded bg-green-500/15 px-1 py-0.5 text-[10px] font-medium text-green-600 dark:text-green-400">
                      In stock
                    </span>
                  ) : o.available === false ? (
                    <span className="shrink-0 rounded bg-muted px-1 py-0.5 text-[10px] font-medium text-muted-foreground">
                      Out of stock
                    </span>
                  ) : null}
                </div>
              </div>
              <div className="flex shrink-0 items-center gap-2">
                {o.price ? (
                  <span className="text-sm font-semibold tabular-nums text-foreground">
                    {o.price}
                  </span>
                ) : null}
                <ExternalLink className="h-4 w-4 text-muted-foreground/60 transition-colors group-hover:text-foreground" />
              </div>
            </a>
          ))}
        </div>
      )}
      {offers.length > 0 && data?.note ? (
        <p className="mt-2 text-[11px] text-muted-foreground/70">{data.note}</p>
      ) : null}
    </div>
  );
}

interface BookingOption {
  platform: string;
  title?: string;
  url: string;
  snippet?: string;
}

interface BookingData {
  query?: string;
  category?: string;
  region?: string;
  options?: BookingOption[];
  note?: string;
}

const CATEGORY_LABEL: Record<string, string> = {
  flight: "Flights",
  hotel: "Hotels",
  movie: "Movies",
  concert: "Concerts",
  event: "Events",
  show: "Shows",
};

export function BookingCards({ data }: { data: BookingData }) {
  const options = (data?.options ?? []).filter((o) => o?.url);
  const label = data?.category
    ? CATEGORY_LABEL[data.category] || "Booking options"
    : "Booking options";

  return (
    <div className="mx-auto w-full max-w-3xl">
      <CardHeader
        icon={<Ticket className="h-4 w-4 text-muted-foreground" />}
        title={label}
        region={data?.region}
      />
      {options.length === 0 ? (
        <p className="rounded-lg border border-dashed border-border px-3 py-4 text-center text-xs text-muted-foreground">
          {data?.note || "No options found, try adding a city, date, or exact title."}
        </p>
      ) : (
        <div className="grid gap-2">
          {options.map((o, i) => (
            <a
              key={`${o.url}-${i}`}
              href={o.url}
              target="_blank"
              rel="noopener noreferrer"
              className="group flex items-start gap-3 rounded-lg border border-border bg-background p-3 transition-colors hover:border-foreground/30 hover:bg-muted/40"
            >
              <Favicon domain={o.platform} />
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-medium text-foreground">
                  {o.title || prettyDomain(o.platform)}
                </div>
                {o.snippet ? (
                  <div className="line-clamp-2 text-xs text-muted-foreground">
                    {o.snippet}
                  </div>
                ) : null}
                <div className="mt-0.5 truncate text-[11px] text-muted-foreground/70">
                  {prettyDomain(o.platform)}
                </div>
              </div>
              <ExternalLink className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground/60 transition-colors group-hover:text-foreground" />
            </a>
          ))}
        </div>
      )}
      {options.length > 0 && data?.note ? (
        <p className="mt-2 text-[11px] text-muted-foreground/70">{data.note}</p>
      ) : null}
    </div>
  );
}
