use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Data, DeriveInput, Fields, Lit, Meta};

/// Derive macro that generates a companion `*Timeseries` struct for collecting
/// per-timestep flux values. All fields in the source struct must be `f64`.
///
/// The generated timeseries struct has the same fields but as `Vec<f64>`,
/// along with `with_capacity`, `push`, `len`, and `is_empty` methods.
/// A `field_names()` associated function is also added to the original struct.
///
/// Use `#[fluxes(timeseries_name = "CustomName")]` to override the default
/// timeseries struct name (`{StructName}Timeseries`).
#[proc_macro_derive(Fluxes, attributes(fluxes))]
pub fn derive_fluxes(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = &input.ident;

    let ts_name = extract_timeseries_name(&input)
        .unwrap_or_else(|| format_ident!("{}Timeseries", name));

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(named) => &named.named,
            _ => {
                return syn::Error::new_spanned(
                    name,
                    "Fluxes can only be derived for structs with named fields",
                )
                .to_compile_error()
                .into();
            }
        },
        _ => {
            return syn::Error::new_spanned(name, "Fluxes can only be derived for structs")
                .to_compile_error()
                .into();
        }
    };

    if fields.is_empty() {
        return syn::Error::new_spanned(name, "Fluxes struct must have at least one field")
            .to_compile_error()
            .into();
    }

    let mut field_names = Vec::new();
    let mut field_idents = Vec::new();
    for field in fields {
        let ident = field.ident.as_ref().unwrap();
        if !is_f64_type(&field.ty) {
            return syn::Error::new_spanned(
                &field.ty,
                "Fluxes derive: all fields must be f64",
            )
            .to_compile_error()
            .into();
        }
        field_names.push(ident.to_string());
        field_idents.push(ident);
    }

    let first_field = &field_idents[0];

    let field_name_strs: Vec<&str> = field_names.iter().map(|s| s.as_str()).collect();

    let ts_fields = field_idents.iter().map(|f| {
        quote! { pub #f: Vec<f64> }
    });

    let with_cap_fields = field_idents.iter().map(|f| {
        quote! { #f: Vec::with_capacity(n) }
    });

    let push_fields = field_idents.iter().map(|f| {
        quote! { self.#f.push(f.#f); }
    });

    let expanded = quote! {
        /// Auto-generated timeseries struct for collecting per-timestep fluxes.
        #[derive(Debug)]
        pub struct #ts_name {
            #(#ts_fields,)*
        }

        impl #ts_name {
            /// Pre-allocate all vectors for `n` timesteps.
            pub fn with_capacity(n: usize) -> Self {
                Self {
                    #(#with_cap_fields,)*
                }
            }

            /// Push a single timestep's fluxes.
            pub fn push(&mut self, f: &#name) {
                #(#push_fields)*
            }

            /// Number of timesteps stored.
            pub fn len(&self) -> usize {
                self.#first_field.len()
            }

            /// Returns `true` if no timesteps have been stored.
            pub fn is_empty(&self) -> bool {
                self.#first_field.is_empty()
            }
        }

        impl #name {
            /// Returns the field names of this flux struct.
            pub fn field_names() -> &'static [&'static str] {
                &[#(#field_name_strs),*]
            }
        }
    };

    expanded.into()
}

fn extract_timeseries_name(input: &DeriveInput) -> Option<proc_macro2::Ident> {
    for attr in &input.attrs {
        if attr.path().is_ident("fluxes") {
            let nested = attr
                .parse_args_with(
                    syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
                )
                .ok()?;
            for meta in nested {
                if let Meta::NameValue(nv) = meta {
                    if nv.path.is_ident("timeseries_name") {
                        if let syn::Expr::Lit(expr_lit) = &nv.value {
                            if let Lit::Str(lit_str) = &expr_lit.lit {
                                return Some(format_ident!("{}", lit_str.value()));
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

fn is_f64_type(ty: &syn::Type) -> bool {
    if let syn::Type::Path(type_path) = ty {
        type_path.path.is_ident("f64")
    } else {
        false
    }
}
